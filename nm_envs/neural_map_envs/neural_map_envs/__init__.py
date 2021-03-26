from gym.envs.registration import register

register(
    id='two_indicators_goals_dimensions-v0',
    entry_point='neural_map_envs.envs:Two_Goals_Indicators_Dimensions',
    kwargs={'map_name'  : '/mnt/ma_neural_map/nm_envs/neural_map_envs/neural_map_envs/envs/map_data_files/EasyMap001.npy',
            'view_straightforward'  : 3,
            'view_left_right'       : 1,
            'start_position_x'  : 2,
            'start_position_y'  : 1,
            'start_orientation' : 0,
            'reward_range'  : (-1., 1.)},
)

register(
    id='one_key_three_boxes_2d-v0',
    entry_point='neural_map_envs.envs:One_Key_Three_Boxes_2D',
    kwargs={'map_name'  : '/mnt/ma_neural_map/nm_envs/neural_map_envs/neural_map_envs/envs/map_data_files/OneKeyThreeBoxes2D.npy',
            'view_straightforward'  : 3,
            'view_left_right'       : 1,
            'start_position_x'  : 3,
            'start_position_y'  : 3,
            'start_orientation' : 0,
            'reward_range'  : (-1., 1.)},
)
