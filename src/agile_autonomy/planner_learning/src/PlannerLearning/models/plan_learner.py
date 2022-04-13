import os

import numpy as np
import tensorflow as tf
from tqdm import tqdm

try:
    # Ros runtime
    from .nets import create_network
    from .data_loader import create_dataset
    from .utils import MixtureSpaceLoss, TrajectoryCostLoss
except:
    # Training time
    from nets import create_network
    from data_loader import create_dataset
    from utils import MixtureSpaceLoss, TrajectoryCostLoss, convert_to_trajectory, \
            save_trajectories, transformToWorldFrame

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class PlanLearner(object):
    def __init__(self, settings):
        self.data_interface = None
        #参数文件
        self.config = settings
        #设置物理设备
        physical_devices = tf.config.experimental.list_physical_devices('GPU')
        if len(physical_devices) > 0:
            tf.config.experimental.set_memory_growth(physical_devices[0], True)
        #trainable=False，后期不被算法所优化
        self.min_val_loss = tf.Variable(np.inf,
                                        name='min_val_loss',
                                        trainable=False)
        #创建一个网络
        self.network = create_network(self.config)
        #定义损失
        self.space_loss = MixtureSpaceLoss(T=self.config.out_seq_len * 0.1, modes=self.config.modes)
        # need two instances due to pointclouds
        #论文中的代价函数
        self.cost_loss = TrajectoryCostLoss(ref_frame=self.config.ref_frame, state_dim=self.config.state_dim)
        self.cost_loss_v = TrajectoryCostLoss(ref_frame=self.config.ref_frame, state_dim=self.config.state_dim)

        # rate scheduler
        self.learning_rate_fn = tf.keras.experimental.CosineDecayRestarts(
            			1e-3,#初始学习率
            			50000,#衰减的步数
            			1.5,#t_mul，用于推导第i个周期的迭代次数
            			0.75,#m_mul，用于推导第 i 个周期的初始学习率
            			0.01)#alpha，最小学习率值作为初始学习率的一部分
        #使用Adam优化器，一种梯度下降算法
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate_fn)
        #计算给定值的（加权）平均值。
        self.train_space_loss = tf.keras.metrics.Mean(name='train_space_loss')
        self.val_space_loss = tf.keras.metrics.Mean(name='validation_space_loss')
        self.train_cost_loss = tf.keras.metrics.Mean(name='train_cost_loss')
        self.val_cost_loss = tf.keras.metrics.Mean(name='validation_cost_loss')
        #创建一个初始化为0的Tensor变量
        self.global_epoch = tf.Variable(0)
        #该类可以进行模型的保存和恢复，保存或恢复step、optimizer、net
        self.ckpt = tf.train.Checkpoint(step=self.global_epoch,
                                        optimizer=self.optimizer,
                                        net=self.network)
        #选择是否使用已存在的训练好的模型
        if self.config.resume_training:
            if self.ckpt.restore(self.config.resume_ckpt_file):
                print("------------------------------------------")
                print("Restored from {}".format(self.config.resume_ckpt_file))
                print("------------------------------------------")
                return

        print("------------------------------------------")
        print("Initializing from scratch.")
        print("------------------------------------------")
    #更新损失函数的过程
    @tf.function
    #输入数据和标签
    def train_step(self, inputs, labels):
        with tf.GradientTape() as tape:
            #inputs是输入的数据
            #predictions为网络的输出output，应该为输出的轨迹
            predictions = self.network(inputs)
            #两个损失函数
            #预测值与真值一起计算相关损失函数，作为训练结果的评判指标
            space_loss = self.space_loss(labels, predictions)
            #和点云信息一起计算的代价函数
            cost_loss = self.cost_loss((inputs['roll_id'], inputs['imu'][:, -1, :12]), predictions)
            loss = space_loss + cost_loss
        #求导，用梯度下降法找到最小的损失函数值
        gradients = tape.gradient(loss, self.network.trainable_variables)
        gradients = [tf.clip_by_norm(g, 1) for g in gradients]
        self.optimizer.apply_gradients(zip(gradients, self.network.trainable_variables))
        #累计指标的统计信息
        #对所有的损失加权平均
        self.train_space_loss.update_state(space_loss)
        self.train_cost_loss.update_state(cost_loss)
        return gradients

    @tf.function
    def val_step(self, inputs, labels):#, epoch, step):
        """
        Perform validation step.
        """

        predictions = self.network(inputs)
        space_loss = self.space_loss(labels, predictions)
        cost_loss = self.cost_loss_v((inputs['roll_id'], inputs['imu'][:, -1, :12]), predictions)
        self.val_space_loss.update_state(space_loss)
        self.val_cost_loss.update_state(cost_loss)

        return predictions

    #网络输入，以及是否使用rgb或depth
    #输入为特征
    def adapt_input_data(self, features):
        if self.config.use_rgb and self.config.use_depth:
            inputs = {"rgb": features[1][0],
                      "depth": features[1][1],
                      "roll_id": features[2],
                      "imu": features[0]}
        elif self.config.use_rgb and (not self.config.use_depth):
            inputs = {"rgb": features[1],
                      "roll_id": features[2],
                      "imu": features[0]}
        elif self.config.use_depth and (not self.config.use_rgb):
            inputs = {"depth": features[1],
                      "roll_id": features[2],
                      "imu": features[0]}
        else:
            inputs = {"imu": features[0],
                      "roll_id": features[1]}
        return inputs

    #记录损失函数
    def write_train_summaries(self, features, gradients):
        with self.summary_writer.as_default():
            #train_space_loss.result()代表输出加权平均后的值
            #step是一个epoch
            tf.summary.scalar('Train Space Loss', self.train_space_loss.result(),
                              step=self.optimizer.iterations)
            tf.summary.scalar('Train Traj_Cost Loss', self.train_cost_loss.result(),
                              step=self.optimizer.iterations)
            # Feel free to add more :)

    def train(self):
        print("Training Network")
        #hasattr() 函数用于判断对象是否包含对应的属性。
        #hasattr(object, name)
        #没有一个训练数据记录的文件夹就创建一个
        if not hasattr(self, 'train_log_dir'):
            # This should be done only once
            #在参数文件中可修改
            self.train_log_dir = os.path.join(self.config.log_dir, 'train')
            #为给定的日志目录创建一个摘要文件编写器
            #logdir：一个字符串，指定写入事件文件的目录
            self.summary_writer = tf.summary.create_file_writer(self.train_log_dir)
            #通过保留一些模型并删除不需要的模型来管理多个模型
            #checkpoint：tf.train.Checkpoint要为其保存和管理检查点的实例。
            #directory：写入检查点的目录的路径
            #max_to_keep：要保留的ckpt文件，大于这个数之后会删除旧的
            self.ckpt_manager = tf.train.CheckpointManager(self.ckpt,
                                                           self.train_log_dir, max_to_keep=20)
        else:
            # We are in dagger mode, so let us reset the best loss
            #如果有文件夹，则说明我们在dagger模式中，重置最佳损失
            #重置所有指标状态变量。当在训练期间评估度量时，在 epochs/steps 之间调用此函数。
            self.min_val_loss = np.inf
            self.train_space_loss.reset_states()
            self.val_space_loss.reset_states()
        #创建训练集
        dataset_train = create_dataset(self.config.train_dir,
                                       self.config, training=True)
        #创建测试集
        dataset_val = create_dataset(self.config.val_dir,
                                     self.config, training=True)  # training was False  Changed by cl to True

        # add pointclouds to losses
        #用于后续的损失计算
        self.cost_loss.add_pointclouds(dataset_train.pointclouds)
        self.cost_loss_v.add_pointclouds(dataset_val.pointclouds)

        for epoch in range(self.config.max_training_epochs):
            # Train
            # Set learning_phase for keras (1 is train, 0 is test)
            # freeze_backbone的值为True则是测试，false则是训练
            if self.config.freeze_backbone:
                tf.keras.backend.set_learning_phase(0)
            else:
                tf.keras.backend.set_learning_phase(1)
            for k, (features, label, _) in enumerate(tqdm(dataset_train.batched_dataset)):
                features = self.adapt_input_data(features)
                gradients = self.train_step(features, label)
                if tf.equal(k % self.config.summary_freq, 0):
                    self.write_train_summaries(features, gradients)
                    self.train_space_loss.reset_states()
                    self.train_cost_loss.reset_states()
            # Eval
            tf.keras.backend.set_learning_phase(0)
            #Tqdm 是一个快速，可扩展的Python进度条，可以在 Python 长循环中添加一个进度提示信息
            #enumerate()函数是python的内置函数，可以同时遍历 lt 中的元素及其索引
            #功能就是读取数据集里的特征信息和标签，K代表的只是第几个特征和标签，从0开始
            for k, (features, label, _) in enumerate(tqdm(dataset_val.batched_dataset)):
                #设置inputs，包含depth和imu
                features = self.adapt_input_data(features)
                #验证损失
                self.val_step(features, label)
            #得出测试集的损失
            val_space_loss = self.val_space_loss.result()
            val_cost_loss = self.val_cost_loss.result()
            validation_loss = val_space_loss + val_cost_loss
            with self.summary_writer.as_default():
                #写一个标量摘要
                #写入到当前的默认摘要编写器。每个汇总点都与一个整step数值相关联。这启用了时间序列数据的增量记录。
                #此 API 的常见用法是在训练期间记录损失以生成损失曲线。
                #name（"Validation Space Loss"）：此摘要的名称
                #data（val_space_loss）：一个实数值标量值，可转换为float32张量。
                #step：单调步长值，相当于曲线横坐标之间的差值
                tf.summary.scalar("Validation Space Loss", val_space_loss,
                                  step=tf.cast(self.global_epoch, dtype=tf.int64))
                #tf.cast
                #将x（self.global_epoch），转换为dtype类型                  
                tf.summary.scalar("Validation Cost Loss", val_cost_loss,
                                  step=tf.cast(self.global_epoch, dtype=tf.int64))
            #重置所有指标状态变量。用于计算下一个epoch的损失
            self.val_space_loss.reset_states()
            self.val_cost_loss.reset_states()

            self.global_epoch = self.global_epoch + 1
            self.ckpt.step.assign_add(1)
            #显示每次的训练结果
            print("Epoch: {:2d}, Val Space Loss: {:.4f}, Val Cost Loss: {:.4f}".format(
                self.global_epoch, val_space_loss, val_cost_loss))
            #当损失满足要求时，保存模型
            if validation_loss < self.min_val_loss or ((epoch + 1) % self.config.save_every_n_epochs) == 0:
                if validation_loss < self.min_val_loss:
                    self.min_val_loss = validation_loss
                if validation_loss < 10.0: # otherwise training diverged
                    #保存模型
                    save_path = self.ckpt_manager.save()
                    print("Saved checkpoint for epoch {}: {}".format(int(self.ckpt.step), save_path))

        print("------------------------------")
        print("Training finished successfully")
        print("------------------------------")
    #这个测试网络是不是就是加载网络的过程
    def test(self):
        print("Testing Network")
        self.train_log_dir = os.path.join(self.config.log_dir, 'test')
        dataset_val = create_dataset(self.config.test_dir,
                                     self.config, training=False)
        self.cost_loss_v.add_pointclouds(dataset_val.pointclouds)
        #mode似乎是用来选择
        if self.config.mode == 'loss':
            tf.keras.backend.set_learning_phase(0)
            for k, (features, label, _) in enumerate(tqdm(dataset_val.batched_dataset)):
                features = self.adapt_input_data(features)
                self.val_step(features, label)
            val_space_loss = self.val_space_loss.result()
            val_cost_loss = self.val_cost_loss.result()
            self.val_space_loss.reset_states()
            self.val_cost_loss.reset_states()
            print("Testing Space Loss: {:.4f} Testing Cost Loss: {:.4f}".format(val_space_loss, val_cost_loss))
         #mode == 'prediction'好像是生成轨迹
        elif self.config.mode == 'prediction':

            for features, label, traj_num in tqdm(dataset_val.batched_dataset):
                features = self.adapt_input_data(features)
                prediction = self.full_post_inference(features)

                trajectories = convert_to_trajectory(label,
                                                     state=features['imu'].numpy()[:, -1, :],
                                                     config=self.config,
                                                     network=False)
                save_trajectories(folder=self.config.log_dir,
                                  trajectories=trajectories,
                                  sample_num=traj_num.numpy())

    def inference(self, inputs):
        # run time inference
        processed_pred = self.full_post_inference(inputs).numpy()
        # Assume BS = 1
        processed_pred = processed_pred[:, np.abs(processed_pred[0, :, 0]).argsort(), :]
        alphas = np.abs(processed_pred[0, :, 0])
        predictions = processed_pred[0, :, 1:]
        return alphas, predictions

    @tf.function
        #返回预测的轨迹
    #input包含的内容可能就是网络的输入
    #按照上面的使用，包含的adapt_input_data函数所设置的depth，和imu信息
    def full_post_inference(self, inputs):
        predictions = self.network(inputs)
        return predictions
