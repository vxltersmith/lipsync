from infer_wav2lip_marlin import main as wav2lip_marlin_gen
from glob import glob
from os.path import join
import os


def main():
    model_configs = [
        {
            "checkpoint_path": "./checkpoints/wav2lip_marlin/audio_emb_lrsmain_l1_id_sync_spec_norm_drop/l1_vl_checkpoint_epoch=116_step=335205.ckpt",
            "wav2lip_type": "marlin",
            "audio_encoder_type": "wav2vec", 
            'model_args': {
                'dropout': 0.05,
                'spectral_scale': True,
                'mean_context': False,
                'resize_to_gen': False,
                'deep_fusion': True
            }
        },
        # {
        #     "checkpoint_path": "./checkpoints/wav2lip_marlin/clean_retrain_lrsmain_l1_id_sync_specnorm_drop/l1_tr_checkpoint_epoch\=102_step\=295095.ckpts",
        #     "wav2lip_type": "marlin",
        #     "audio_encoder_type": "default", 
        #     'model_args': {
        #         'dropout': 0.05,
        #         'spectral_scale': True,
        #         'mean_context': False,
        #         'resize_to_gen': False,
        #         'deep_fusion': False
        #     }
        # },
        # {
        #     "checkpoint_path": "./checkpoints/wav2lip_marlin/l1_sl_idl_lrs_pretrain_main/l1_vl_checkpoint_epoch=102_step=295095.ckpt",
        #     "wav2lip_type": "marlin",
        #     "audio_encoder_type": "default", 
        #     'model_args': {
        #         'dropout': 0.05,
        #         'spectral_scale': True,
        #         'mean_context': False,
        #         'resize_to_gen': False,
        #         'deep_fusion': False
        #     }
        # },
        # {
        #     "checkpoint_path": "./checkpoints/wav2lip_marlin/clean_retrain_lrsmain_l1_id_sync_specnorm_drop_continue/l1_vl_checkpoint_epoch=106_step=306555.ckpt",
        #     "wav2lip_type": "marlin",
        #     "audio_encoder_type": "default", 
        #     'model_args': {
        #         'dropout': 0.05,
        #         'spectral_scale': True,
        #         'mean_context': False,
        #         'resize_to_gen': False,
        #         'deep_fusion': False
        #     }
        # },
    ]
    
    output_dir = './toloka_test_v0/generated_best_videos'
    target_video_path = None
    gif_dir = 'gifs'
    start_at = 3
    
    # get list of videos
    source_videos_dir = './toloka_test_v0/monologues_renamed_25fps/'
    source_videos = list(glob(join(source_videos_dir, '*.mp4')))
    
    # get list of audios
    target_audios_dir = './toloka_test_v0/audio_targets/'
    target_audios = list(glob(join(target_audios_dir, '*.*')))
    
    for model_config in model_configs:
        checkpoint_path = model_config['checkpoint_path']
        if not os.path.exists(checkpoint_path): continue
                
        output_sub_dir = os.path.join(output_dir, os.path.dirname(checkpoint_path).split('/')[-1])
        
        audio_encoder_type = model_config['audio_encoder_type']
        wav2lip_type = model_config['wav2lip_type']
        model_args = model_config['model_args']
        
        for source_video_path in source_videos:
            for audio_path in target_audios:
                
                wav2lip_marlin_gen(source_video_path, audio_path=audio_path, target_video_path=target_video_path, 
                    audio_encoder_type=audio_encoder_type, start_at=start_at, output_dir=output_sub_dir, 
                    wav2lip_type=wav2lip_type, checkpoint_path=checkpoint_path, gif_dir=gif_dir,
                    model_args = model_args
                )
    return

if __name__ == "__main__":
    main()