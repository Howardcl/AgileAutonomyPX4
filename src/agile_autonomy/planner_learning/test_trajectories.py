import argparse
from config.settings import create_settings
from dagger_training import Trainer


def main():
    # parser = argparse.ArgumentParser(description='Evaluate Trajectory tracker.')
    # parser.add_argument('--settings_file', help='Path to settings yaml', required=True)
    # args = parser.parse_args()
    # settings_filepath = args.settings_file

    #直接给定参数文件的绝对路径去加载
    settings_filepath = '/home/pc205/agile_autonomy_ws/catkin_aa/src/agile_autonomy/planner_learning/config/test_settings.yaml' 
    settings = create_settings(settings_filepath, mode='test')

    settings_filepath_2 = '/home/pc205/agile_autonomy_ws/catkin_aa/src/agile_autonomy/planner_learning/config/test_settings_2.yaml' 
    settings_2 = create_settings(settings_filepath_2, mode='test')

    trainer = Trainer(settings)
    trainer_2 = Trainer(settings_2)
    trainer.perform_testing()
    trainer_2.perform_testing()


if __name__ == "__main__":
    main()
