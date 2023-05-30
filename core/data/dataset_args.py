from configs import cfg

class DatasetArgs(object):
    dataset_attrs = {}
    
    if cfg.category == 'human_nerf' and cfg.task == 'wild':
        dataset_attrs.update({
            "monocular_train": {
                "dataset_path": f'processed_data/{cfg.subject}',
                "keyfilter": cfg.train_keyfilter,
                "ray_shoot_mode": cfg.train.ray_shoot_mode,
            },
            "monocular_test": {
                "dataset_path": f'processed_data/{cfg.subject}',  
                "keyfilter": cfg.test_keyfilter,
                "ray_shoot_mode": 'image',
                "src_type": 'wild'
            },
        })


    @staticmethod
    def get(name):
        attrs = DatasetArgs.dataset_attrs[name]
        return attrs.copy()
