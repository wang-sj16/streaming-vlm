import os
import os.path as osp
import pandas as pd
import numpy as np
from ..smp import *
from .video_base import VideoBaseDataset
from .utils import build_judge, DEBUG_MESSAGE

FAIL_MSG = 'Failed to obtain answer via API.'

# Default path for LVBench dataset
LVBENCH_ROOT = '/oscar/data/csun45/bli169/Stream/datasets/LVBench'


class LVBench(VideoBaseDataset):
    """
    LVBench: An Extreme Long Video Understanding Benchmark
    
    Dataset format:
    - video_path: path to video file (e.g., "Cm73ma6Ibcs.mp4")
    - uid: unique question id
    - question: question text with options (A/B/C/D format)
    - question_type: list of question types
    - answer: correct answer (A/B/C/D)
    - time_reference: time range for the question
    - type: video type (cartoon, etc.)
    - key: video key/id
    """
    
    TYPE = 'Video-MCQ'
    SYS = ''
    
    def __init__(self, dataset='LVBench', nframe=0, fps=-1, root=None):
        self.custom_root = root if root else LVBENCH_ROOT
        super().__init__(dataset=dataset, nframe=nframe, fps=fps)
        self.dataset_name = dataset
    
    @classmethod
    def supported_datasets(cls):
        return ['LVBench']
    
    def prepare_dataset(self, dataset_name='LVBench', repo_id='THUDM/LVBench'):
        """
        Prepare the LVBench dataset.
        Uses local path instead of downloading from HuggingFace.
        """
        cache_path = self.custom_root
        
        def check_integrity(pth):
            data_file = osp.join(pth, f'{dataset_name}.tsv')
            if not osp.exists(data_file):
                return False
            # Check if video directory exists
            video_dir = osp.join(pth, 'video_chunks')
            if not osp.exists(video_dir):
                return False
            return True
        
        def generate_tsv(pth):
            """Generate TSV file from parquet data"""
            data_file = osp.join(pth, f'{dataset_name}.tsv')
            if osp.exists(data_file):
                return
            
            # Read parquet file
            parquet_file = osp.join(pth, 'data', 'train-00000-of-00001.parquet')
            if not osp.exists(parquet_file):
                raise FileNotFoundError(f"Parquet file not found: {parquet_file}")
            
            df = pd.read_parquet(parquet_file)
            
            # Add index column
            df['index'] = range(len(df))
            
            # Create video column (without .mp4 extension for compatibility)
            df['video'] = df['video_path'].apply(lambda x: x.replace('.mp4', ''))
            
            # Update video_path to include the video_chunks directory
            df['video_path'] = df['video_path'].apply(lambda x: f'video_chunks/{x}')
            
            # Convert question_type numpy array to string for TSV storage
            df['question_type'] = df['question_type'].apply(lambda x: str(list(x)) if hasattr(x, 'tolist') else str(x))
            
            # Save as TSV
            df.to_csv(data_file, sep='\t', index=False)
            print(f"Generated TSV file: {data_file}")
        
        if not check_integrity(cache_path):
            generate_tsv(cache_path)
        
        if not check_integrity(cache_path):
            raise RuntimeError(f"LVBench dataset not properly set up at {cache_path}")
        
        data_file = osp.join(cache_path, f'{dataset_name}.tsv')
        return dict(data_file=data_file, root=cache_path)
    
    def save_video_frames(self, video_path, video_llm=False):
        """Extract and save video frames"""
        vid_path = osp.join(self.data_root, video_path)
        
        import decord
        vid = decord.VideoReader(vid_path)
        video_info = {
            'fps': vid.get_avg_fps(),
            'n_frames': len(vid),
        }
        
        # When video_llm=True, model processes video directly, no need to extract frames
        if video_llm:
            return [], [], video_info
        
        # Get video name without extension
        video_name = video_path.replace('video_chunks/', '').replace('.mp4', '')
        
        if self.nframe > 0 and self.fps < 0:
            step_size = len(vid) / (self.nframe + 1)
            indices = [int(i * step_size) for i in range(1, self.nframe + 1)]
            frame_paths = self.frame_paths(video_name)
        elif self.fps > 0:
            total_duration = video_info['n_frames'] / video_info['fps']
            required_frames = int(total_duration * self.fps)
            step_size = video_info['fps'] / self.fps
            indices = [int(i * step_size) for i in range(required_frames)]
            frame_paths = self.frame_paths_fps(video_name, len(indices))
        
        flag = np.all([osp.exists(p) for p in frame_paths])
        
        if not flag:
            lock_path = osp.splitext(vid_path)[0] + '.lock'
            max_retries = 5
            for retry in range(max_retries):
                try:
                    with portalocker.Lock(lock_path, 'w', timeout=120):
                        if not np.all([osp.exists(p) for p in frame_paths]):
                            images = [vid[i].asnumpy() for i in indices]
                            images = [Image.fromarray(arr) for arr in images]
                            for im, pth in zip(images, frame_paths):
                                if not osp.exists(pth):
                                    im.save(pth)
                    break
                except portalocker.LockException as e:
                    if retry < max_retries - 1:
                        import time
                        import random
                        time.sleep(random.uniform(1, 5))
                    else:
                        raise e
        
        return frame_paths, indices, video_info
    
    def build_prompt(self, line, video_llm=False):
        """Build the prompt for a single question"""
        if isinstance(line, int):
            assert line < len(self)
            line = self.data.iloc[line]
        
        frames, indices, video_info = self.save_video_frames(line['video_path'], video_llm)
        
        message = [dict(type='text', value=self.SYS)] if self.SYS else []
        
        if video_llm:
            message.append(dict(type='video', value=osp.join(self.data_root, line['video_path'])))
        else:
            for im in frames:
                message.append(dict(type='image', value=im))
        
        # The question already contains options in (A)/(B)/(C)/(D) format
        question = line['question']
        prompt = question + "\nAnswer with the option's letter from the given choices directly."
        message.append(dict(type='text', value=prompt))
        
        return message
    
    @classmethod
    def evaluate(cls, eval_file, **judge_kwargs):
        """Evaluate predictions against ground truth"""
        from .utils.lvbench import get_dimension_rating, extract_characters_regex
        
        assert eval_file.endswith('.xlsx'), 'data file should be an xlsx file'
        
        score_file = eval_file.replace('.xlsx', '_score.xlsx')
        tgt_file = eval_file.replace('.xlsx', '_rating.json')
        
        if not osp.exists(score_file):
            data = load(eval_file)
            data_un = data[~pd.isna(data['prediction'])]
            
            for idx in data['index']:
                ans = data.loc[data['index'] == idx, 'answer'].values[0]
                pred = str(data.loc[data['index'] == idx, 'prediction'].values[0])
                
                # Extract the answer letter from prediction
                extracted = extract_characters_regex(pred)
                data.loc[data['index'] == idx, 'score'] = int(extracted == ans)
            
            rejected = [x for x in data['score'] if x == -1]
            
            print(
                f'Among {len(data)} questions, failed to obtain prediction for {len(data) - len(data_un)} questions, '
                f'failed to obtain the score for another {len(rejected)} questions. '
            )
            
            dump(data, score_file)
        
        rating = get_dimension_rating(score_file)
        dump(rating, tgt_file)
        return rating

