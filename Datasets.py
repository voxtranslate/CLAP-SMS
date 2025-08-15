import torch
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image, ImageFile
import torchvision.transforms as transforms
import random
import numpy as np
import os

# =============================================
# AUDIO PROCESSOR
# =============================================

class AudioProcessor:
    """Audio processor with robust MP3 support and variable-length handling"""
    
    def __init__(self, sr=22050, n_fft=1024, hop_length=256, n_mels=80, 
                 min_length=1.0, max_length=60.0):
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.min_length = min_length
        self.max_length = max_length
        
        # Mel spectrogram transform
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            normalized=True
        )
        
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB(
            stype='magnitude', top_db=80
        )
        
        self._setup_audio_backend()
    
    def _setup_audio_backend(self):
        """Setup proper audio backend for MP3 support"""
        try:
            if torchaudio.get_audio_backend() != 'ffmpeg':
                try:
                    torchaudio.set_audio_backend('ffmpeg')
                    print("Using FFmpeg backend for audio loading")
                except:
                    print("FFmpeg backend not available, using default backend")
            
            available_backends = torchaudio.list_audio_backends()
            print(f"Available audio backends: {available_backends}")
            
        except Exception as e:
            print(f"Audio backend setup warning: {e}")
    
    def validate_audio(self, audio_path):
        """Validate audio file and return basic info"""
        try:
            info = torchaudio.info(str(audio_path))
            duration = info.num_frames / info.sample_rate
        except Exception:
            try:
                y, sr = librosa.load(str(audio_path), sr=None, duration=1.0)
                duration = librosa.get_duration(path=str(audio_path))
                info = type('Info', (), {
                    'num_frames': int(duration * sr),
                    'sample_rate': sr,
                    'num_channels': 1 if y.ndim == 1 else y.shape[0]
                })
            except Exception as e2:
                return False, f"Both torchaudio and librosa failed: {e2}"
        
        if duration < 0.05:
            return False, f"Duration {duration:.3f}s too short (minimum 0.05s)"
        
        if duration > 1800:
            return False, f"Duration {duration:.1f}s too long (maximum 1800s)"
        
        if info.sample_rate < 4000:
            return False, f"Sample rate {info.sample_rate} too low (minimum 4000Hz)"
        
        if hasattr(info, 'num_channels') and info.num_channels > 8:
            return False, f"Too many channels: {info.num_channels} (maximum 8)"
        
        return True, {
            "duration": duration, 
            "sr": info.sample_rate, 
            "channels": getattr(info, 'num_channels', 1)
        }
    
    def load_and_preprocess(self, audio_path, target_length=None, augment=False):
        """Load and preprocess audio with robust length handling"""
        waveform, orig_sr = self._load_audio_robust(audio_path)
        
        # Resample if needed
        if orig_sr != self.sr:
            resampler = torchaudio.transforms.Resample(orig_sr, self.sr)
            waveform = resampler(waveform)
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Normalize to [-1, 1] range
        max_val = torch.max(torch.abs(waveform))
        if max_val > 0:
            waveform = waveform / max_val
        
        # Trim silence from beginning and end
        waveform = self._trim_silence(waveform)
        
        # Apply length constraints
        min_samples = int(self.sr * max(self.min_length, 3.0))
        max_samples = int(self.sr * self.max_length)
        
        # Handle too short audio by repeating
        if waveform.shape[1] < min_samples:
            repeat_factor = (min_samples // waveform.shape[1]) + 1
            waveform = waveform.repeat(1, repeat_factor)
            waveform = waveform[:, :min_samples]
        
        # Handle too long audio by cropping from center
        if waveform.shape[1] > max_samples:
            start = (waveform.shape[1] - max_samples) // 2
            waveform = waveform[:, start:start + max_samples]
        
        # Apply target length if specified
        if target_length is not None:
            target_samples = int(self.sr * max(target_length, 3.0))
            current_samples = waveform.shape[1]
            
            if current_samples != target_samples:
                if current_samples > target_samples:
                    start = (current_samples - target_samples) // 2
                    waveform = waveform[:, start:start + target_samples]
                else:
                    pad_amount = target_samples - current_samples
                    waveform = F.pad(waveform, (0, pad_amount))
        
        actual_length = waveform.shape[1] / self.sr
        mel_spec = self.compute_mel_spectrogram(waveform)
        
        # Ensure minimum mel spectrogram dimensions
        if mel_spec.shape[0] < 8 or mel_spec.shape[1] < 10:
            target_freq = max(mel_spec.shape[0], 8)
            target_time = max(mel_spec.shape[1], 10)
            mel_spec = F.pad(mel_spec, 
                           (0, target_time - mel_spec.shape[1], 
                            0, target_freq - mel_spec.shape[0]), 
                           mode='reflect')
        
        return mel_spec, actual_length, True, waveform
    
    def _load_audio_robust(self, audio_path):
        """Robust audio loading with multiple fallback methods"""
        audio_path = str(audio_path)
        
        try:
            waveform, sample_rate = torchaudio.load(audio_path)
            return waveform, sample_rate
        except Exception:
            try:
                y, sr = librosa.load(audio_path, sr=None, mono=False)
                if y.ndim == 1:
                    waveform = torch.from_numpy(y).unsqueeze(0).float()
                else:
                    waveform = torch.from_numpy(y).float()
                return waveform, sr
            except Exception:
                y, sr = librosa.load(audio_path, sr=22050, mono=True)
                waveform = torch.from_numpy(y).unsqueeze(0).float()
                return waveform, sr
    
    def _trim_silence(self, waveform, threshold=0.01):
        """Trim silence from beginning and end"""
        energy = torch.sum(waveform ** 2, dim=0)
        above_threshold = energy > threshold * torch.max(energy)
        
        if torch.any(above_threshold):
            nonzero_indices = torch.nonzero(above_threshold, as_tuple=False).squeeze()
            if nonzero_indices.numel() > 0:
                if nonzero_indices.dim() == 0:
                    start_idx = end_idx = nonzero_indices.item()
                else:
                    start_idx = nonzero_indices[0].item()
                    end_idx = nonzero_indices[-1].item()
                
                padding = int(0.1 * self.sr)
                start_idx = max(0, start_idx - padding)
                end_idx = min(waveform.shape[1], end_idx + padding)
                
                waveform = waveform[:, start_idx:end_idx]
        
        return waveform
    
    def compute_mel_spectrogram(self, waveform):
        """Compute mel spectrogram with proper normalization"""
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        
        min_samples = self.hop_length * 10
        if waveform.shape[1] < min_samples:
            repeat_factor = (min_samples // waveform.shape[1]) + 1
            waveform = waveform.repeat(1, repeat_factor)
            waveform = waveform[:, :min_samples]
        
        mel_spec = self.mel_transform(waveform)
        mel_spec_db = self.amplitude_to_db(mel_spec)
        
        min_time_steps = 10
        if mel_spec_db.shape[2] < min_time_steps:
            pad_amount = min_time_steps - mel_spec_db.shape[2]
            mel_spec_db = F.pad(mel_spec_db, (0, pad_amount), mode='reflect')
        
        min_freq_bins = 8
        if mel_spec_db.shape[1] < min_freq_bins:
            pad_amount = min_freq_bins - mel_spec_db.shape[1]
            mel_spec_db = F.pad(mel_spec_db, (0, 0, 0, pad_amount), mode='reflect')
        
        mel_spec_db = (mel_spec_db - mel_spec_db.mean()) / (mel_spec_db.std() + 1e-8)
        
        return mel_spec_db.squeeze(0)

# =============================================
# BASE DATASET CLASS - CORRECTED
# =============================================

class BaseAudioDataset(Dataset):
    """Base class for audio datasets with consistent return format"""
    
    def __init__(self, audio_processor, tokenizer, max_samples=None):
        self.audio_processor = audio_processor
        self.tokenizer = tokenizer
        self.max_samples = max_samples
        self.valid_samples = []
        
    def _load_metadata(self):
        """Override in subclasses to load metadata"""
        raise NotImplementedError
    
    def _validate_samples(self):
        """Override in subclasses to validate samples"""
        raise NotImplementedError
    
    def __len__(self):
        return len(self.valid_samples)
    
    def __getitem__(self, idx):
        """Ensure consistent return format across all datasets"""
        sample_data = self._process_sample(idx)
        
        # Ensure all required keys are present
        if 'text' not in sample_data or not sample_data['text']:
            sample_data['text'] = sample_data.get('raw_text', 'Audio sample')
        
        # Process text with tokenizer
        text_inputs = self.tokenizer(
            sample_data['text'],
            return_tensors='pt',
            padding='max_length',
            max_length=128,
            truncation=True
        )
        
        # Get audio features
        audio_features = sample_data.get('audio_features')
        if audio_features is None:
            # Create dummy audio features if missing
            audio_features = torch.randn(80, 100)
        
        # Ensure proper dimensions
        if audio_features.dim() == 1:
            audio_features = audio_features.unsqueeze(0).expand(80, -1)
        elif audio_features.dim() == 3:
            audio_features = audio_features.squeeze(0)
        
        # Ensure correct shape
        if audio_features.shape[0] != 80:
            if audio_features.shape[0] < 80:
                pad_amount = 80 - audio_features.shape[0]
                audio_features = F.pad(audio_features, (0, 0, 0, pad_amount), mode='reflect')
            else:
                audio_features = audio_features[:80, :]
        
        # Get modality label
        modality = sample_data.get('modality', 'sound')
        modality_label = 0 if modality in ['speech'] else 1
        
        # Return consistent format
        return {
            'audio_features': audio_features,
            'text_input_ids': text_inputs['input_ids'].squeeze(0),
            'text_attention_mask': text_inputs['attention_mask'].squeeze(0),
            'raw_text': sample_data['text'],
            'audio_length': sample_data.get('audio_length', 3.0),
            'domain': sample_data.get('domain', 'sound_events'),
            'language': sample_data.get('language', 'en'),
            'dataset': sample_data.get('dataset', 'unknown'),
            'modality': modality,
            'modality_label': modality_label,
            'success': sample_data.get('success', True),
            'metadata': {
                'audio_path': sample_data.get('audio_path', ''),
                'sampling_group': sample_data.get('sampling_group', 'sound_events')
            }
        }
    
    def _process_sample(self, idx):
        """Override in subclasses to process individual samples"""
        raise NotImplementedError
    
    def _create_enhanced_text(self, text, prefix=""):
        """Create enhanced text description with prefix"""
        if prefix:
            return f"{prefix}: {text}"
        return text

# =============================================
# SPECIFIC DATASET IMPLEMENTATIONS - CORRECTED
# =============================================

class LJSpeechDataset(BaseAudioDataset):
    """LJ Speech dataset for CLAMP training"""
    
    def __init__(self, root_dir, audio_processor, tokenizer, max_samples=None, metadata_file='metadata.csv'):
        super().__init__(audio_processor, tokenizer, max_samples)
        self.root_dir = Path(root_dir)
        self.metadata_file = metadata_file
        self._load_and_validate()
        
    def _load_metadata(self):
        metadata_path = self.root_dir / self.metadata_file
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
        
        df = pd.read_csv(metadata_path, sep='|', header=None, 
                        names=['ID', 'Transcription', 'Normalized_Transcription'])
        return df
    
    def _load_and_validate(self):
        metadata = self._load_metadata()
        for idx, row in tqdm(metadata.iterrows(), total=len(metadata), desc="Validating LJ Speech"):
            audio_path = self.root_dir / 'wavs' / f"{row['ID']}.wav"
            if not audio_path.exists():
                continue
            
            text = str(row.get('Normalized_Transcription', row.get('Transcription', '')))
            if len(text.strip()) < 5:
                continue
            
            self.valid_samples.append({
                'audio_path': str(audio_path),
                'text': text.strip(),
                'dataset': 'ljspeech',
                'domain': 'english_speech',
                'language': 'en',
                'modality': 'speech'
            })
            count += 1
        
        if not self.valid_samples:
            raise ValueError("No valid LJ Speech samples found")
        
        print(f"LJ Speech: {len(self.valid_samples)} valid samples")
    
    def _process_sample(self, idx):
        sample = self.valid_samples[idx]
        enhanced_text = self._create_enhanced_text(sample['text'], "Speech")
        
        result = self.audio_processor.load_and_preprocess(sample['audio_path'])
        if len(result) == 4:
            mel_spec, length, success, waveform = result
        else:
            mel_spec, length, success = result
        
        return {
            'audio_features': mel_spec,
            'text': enhanced_text,
            'audio_length': length,
            'domain': sample['domain'],
            'language': sample['language'],
            'dataset': sample['dataset'],
            'modality': sample['modality'],
            'success': success,
            'audio_path': sample['audio_path']
        }

class FMADataset(BaseAudioDataset):
    """FMA dataset with robust MP3 loading"""
    
    def __init__(self, root_dir, audio_processor, tokenizer, subset='small', max_samples=None):
        super().__init__(audio_processor, tokenizer, max_samples)
        self.root_dir = Path(root_dir)
        self.subset = subset
        self._load_and_validate()
        
    def _load_metadata(self):
        metadata_path = self.root_dir / 'fma_metadata' / 'tracks.csv'
        
        if not metadata_path.exists():
            raise FileNotFoundError(f"FMA metadata not found: {metadata_path}")
        
        try:
            tracks = pd.read_csv(metadata_path, index_col=0, header=[0, 1])
            print(f"Loaded FMA metadata from {metadata_path} with multi-level headers")
        except:
            tracks = pd.read_csv(metadata_path, index_col=0)
            print(f"Loaded FMA metadata from {metadata_path} with simple headers")
        
        return tracks
    
    def _find_audio_files(self):
        """Find MP3 files"""
        base_dir = self.root_dir / f'fma_{self.subset}' / f'fma_{self.subset}'
        
        if not base_dir.exists():
            raise FileNotFoundError(f"FMA audio directory not found: {base_dir}")
        mp3_files = list(base_dir.rglob('*.mp3'))
        if not mp3_files:
            raise ValueError(f"No MP3 files found in {base_dir}")
        print(f"Found {len(mp3_files)} MP3 files in {base_dir}")
        return mp3_files
   
    def _load_and_validate(self):
        metadata = self._load_metadata()
        mp3_files = self._find_audio_files()
        for audio_path in mp3_files:
            try:
                track_id = int(audio_path.stem)
                is_valid, info = self.audio_processor.validate_audio(audio_path)
                if not is_valid:
                    continue
                genre, artist, title = self._get_track_info(metadata, track_id)
                self.valid_samples.append({
                   'audio_path': str(audio_path),
                   'track_id': track_id,
                   'genre': genre,
                   'artist': artist,
                   'title': title,
                   'dataset': 'fma',
                   'domain': 'music',
                   'language': 'en',
                   'modality': 'music',
                   'duration': info.get('duration', 0) if isinstance(info, dict) else 0
                })
                count += 1
            except (ValueError, Exception):
                continue
        if not self.valid_samples:
            raise ValueError("No valid FMA samples found")
            
        print(f"FMA {self.subset}: {len(self.valid_samples)} valid samples")
   
    def _get_track_info(self, metadata, track_id):
       """Extract track information"""
       if not metadata.empty and track_id in metadata.index:
           track_info = metadata.loc[track_id]
           try:
               if isinstance(metadata.columns, pd.MultiIndex):
                   genre = str(track_info.get(('track', 'genre_top'), 'Electronic'))
                   artist = str(track_info.get(('artist', 'name'), 'Unknown Artist'))
                   title = str(track_info.get(('track', 'title'), f'Track {track_id}'))
               else:
                   genre = str(track_info.get('genre_top', 'Electronic'))
                   artist = str(track_info.get('artist_name', 'Unknown Artist'))
                   title = str(track_info.get('title', f'Track {track_id}'))
               
               genre = genre.replace('nan', 'Electronic')
               artist = artist.replace('nan', 'Unknown Artist')
               title = title.replace('nan', f'Track {track_id}')
               
           except Exception:
               genre, artist, title = 'Electronic', 'Unknown Artist', f'Track {track_id}'
       else:
           genre, artist, title = 'Electronic', 'Unknown Artist', f'Track {track_id}'
       
       return genre, artist, title
   
    def _process_sample(self, idx):
        sample = self.valid_samples[idx]
        enhanced_text = f"Music: {sample['genre']} genre by {sample['artist']}. Title: {sample['title']}"
        
        result = self.audio_processor.load_and_preprocess(sample['audio_path'])
        if len(result) == 4:
            mel_spec, length, success, waveform = result
        else:
            mel_spec, length, success = result
        
        return {
            'audio_features': mel_spec,
            'text': enhanced_text,
            'audio_length': length,
            'domain': sample['domain'],
            'language': sample['language'],
            'dataset': sample['dataset'],
            'modality': sample['modality'],
            'success': success,
            'audio_path': sample['audio_path']
        }

class CommonVoiceDataset(BaseAudioDataset):
    """Common Voice dataset for multilingual speech"""
    
    def __init__(self, root_dir, audio_processor, tokenizer, max_samples=None, target_languages=None):
        super().__init__(audio_processor, tokenizer, max_samples)
        self.root_dir = Path(root_dir)
        self.target_languages = target_languages
        self._load_and_validate()
        
    def _load_and_validate(self):
        cv_files_mapping = {
            'cv-valid-train.csv': 'cv-valid-train',
            'cv-valid-dev.csv': 'cv-valid-dev', 
            'cv-valid-test.csv': 'cv-valid-test',
            'cv-other-train.csv': 'cv-other-train',
            'cv-other-dev.csv': 'cv-other-dev',
            'cv-other-test.csv': 'cv-other-test'
        }
        
        available_files = []
        for csv_file, audio_folder in cv_files_mapping.items():
            csv_file_path = self.root_dir / csv_file
            audio_folder_path = self.root_dir / audio_folder
            
            if csv_file_path.exists() and audio_folder_path.exists():
                available_files.append((csv_file_path, audio_folder_path, csv_file))
        
        if not available_files:
            raise FileNotFoundError("No Common Voice CSV files and audio folders found")
        
        samples_per_file = self.max_samples // len(available_files) if self.max_samples else None
        
        for csv_file_path, audio_folder_path, csv_name in available_files:
            self._load_csv_data(csv_file_path, audio_folder_path, csv_name, samples_per_file)
        
        if not self.valid_samples:
            raise ValueError("No valid Common Voice samples found")
        
        print(f"Common Voice: {len(self.valid_samples)} valid samples")
    
    def _load_csv_data(self, csv_file_path, audio_folder_path, csv_name, max_samples):
        """Load data from a specific CSV file and its audio folder"""
        metadata = pd.read_csv(csv_file_path)
        
        required_columns = ['filename', 'text']
        if not all(col in metadata.columns for col in required_columns):
            return
        for idx, row in metadata.iterrows():
            filename = row.get('filename', '')
            text = row.get('text', '')
            
            if not filename or pd.isna(text) or len(str(text).strip()) < 5:
                continue
            
            audio_path = audio_folder_path / filename
            if not audio_path.exists():
                for ext in ['.wav', '.mp3', '.flac', '.ogg']:
                    test_path = audio_folder_path / f"{Path(filename).stem}{ext}"
                    if test_path.exists():
                        audio_path = test_path
                        break
                else:
                    continue
            
            language = row.get('locale', row.get('language', 'en'))
            if self.target_languages and not any(lang in language for lang in self.target_languages):
                continue
            
            domain = 'english_speech' if language.startswith('en') else 'multilingual_speech'
            
            self.valid_samples.append({
                'audio_path': str(audio_path),
                'text': str(text).strip(),
                'locale': language,
                'dataset': 'common_voice',
                'domain': domain,
                'language': language.split('-')[0],
                'modality': 'speech',
                'split': csv_name.replace('.csv', '')
            })
            count += 1
    
    def _process_sample(self, idx):
        sample = self.valid_samples[idx]
        
        result = self.audio_processor.load_and_preprocess(sample['audio_path'])
        if len(result) == 4:
            mel_spec, length, success, waveform = result
        else:
            mel_spec, length, success = result
        
        lang_name = self._get_language_name(sample['language'])
        enhanced_text = f"{lang_name} speech: {sample['text']}"
        
        return {
            'audio_features': mel_spec,
            'text': enhanced_text,
            'audio_length': length,
            'domain': sample['domain'],
            'language': sample['language'],
            'dataset': sample['dataset'],
            'modality': sample['modality'],
            'success': success,
            'audio_path': sample['audio_path']
        }
    
    def _get_language_name(self, lang_code):
        lang_names = {
            'en': 'English', 'es': 'Spanish', 'fr': 'French', 'de': 'German',
            'zh': 'Chinese', 'ja': 'Japanese', 'ko': 'Korean', 'ar': 'Arabic',
            'hi': 'Hindi', 'pt': 'Portuguese', 'ru': 'Russian', 'it': 'Italian'
        }
        return lang_names.get(lang_code, lang_code.capitalize())

class ESC50Dataset(BaseAudioDataset):
    """ESC-50 environmental sounds dataset"""
    
    def __init__(self, csv_path, audio_root, audio_processor, tokenizer, max_samples=None):
        super().__init__(audio_processor, tokenizer, max_samples)
        self.csv_path = csv_path
        self.audio_root = Path(audio_root)
        self._load_and_validate()
        
    def _load_and_validate(self):
        if not Path(self.csv_path).exists():
            raise FileNotFoundError(f"ESC-50 CSV not found: {self.csv_path}")
        
        metadata = pd.read_csv(self.csv_path)
        for idx, row in tqdm(metadata.iterrows(), total=len(metadata), desc="Validating ESC-50"):
            filename = row.get('filename', '')
            if not filename:
                continue
                
            audio_path = self.audio_root / 'audio' / 'audio' / filename
            
            if not audio_path.exists():
                continue
            
            self.valid_samples.append({
                'audio_path': str(audio_path),
                'category': row.get('category', 'unknown'),
                'fold': row.get('fold', 1),
                'dataset': 'esc50',
                'domain': 'sound_events',
                'language': 'en',
                'modality': 'sound'
            })
            count += 1
        
        if not self.valid_samples:
            raise ValueError("No valid ESC-50 samples found")
        
        print(f"ESC-50: {len(self.valid_samples)} valid samples")
    
    def _process_sample(self, idx):
        sample = self.valid_samples[idx]
        enhanced_text = self._create_rich_description(sample['category'])
        
        result = self.audio_processor.load_and_preprocess(sample['audio_path'])
        if len(result) == 4:
            mel_spec, length, success, waveform = result
        else:
            mel_spec, length, success = result
        
        return {
            'audio_features': mel_spec,
            'text': enhanced_text,
            'audio_length': length,
            'domain': sample['domain'],
            'language': sample['language'],
            'dataset': sample['dataset'],
            'modality': sample['modality'],
            'success': success,
            'audio_path': sample['audio_path']
        }
    
    def _create_rich_description(self, category):
        descriptions = {
            'dog': 'The sound of a dog barking loudly',
            'rain': 'Heavy rainfall on various surfaces',
            'sea_waves': 'Ocean waves crashing on the shore',
            'baby_cry': 'A baby crying loudly',
            'clock_tick': 'A clock ticking steadily'
        }
        default_desc = f"The sound of {category.replace('_', ' ')}"
        return f"Environmental sound: {descriptions.get(category, default_desc)}"

class UrbanSound8KDataset(BaseAudioDataset):
    """UrbanSound8K dataset for urban sounds"""
    
    def __init__(self, csv_path, audio_root, audio_processor, tokenizer, max_samples=None):
        super().__init__(audio_processor, tokenizer, max_samples)
        self.csv_path = csv_path
        self.audio_root = Path(audio_root)
        self._load_and_validate()
        
    def _load_and_validate(self):
        if not Path(self.csv_path).exists():
            raise FileNotFoundError(f"UrbanSound8K CSV not found: {self.csv_path}")
        
        metadata = pd.read_csv(self.csv_path)
        for idx, row in tqdm(metadata.iterrows(), total=len(metadata), desc="Validating UrbanSound8K"):
            fold = row.get('fold', 1)
            filename = row.get('slice_file_name', '')
            
            if not filename:
                continue
            
            audio_path = self.audio_root / f'fold{fold}' / filename
            
            if not audio_path.exists():
                continue
            
            self.valid_samples.append({
                'audio_path': str(audio_path),
                'class_name': row.get('class', row.get('classID', 'unknown')),
                'fold': fold,
                'dataset': 'urbansound8k',
                'domain': 'sound_events',
                'language': 'en',
                'modality': 'sound'
            })
            count += 1
        
        if not self.valid_samples:
            raise ValueError("No valid UrbanSound8K samples found")
        
        print(f"UrbanSound8K: {len(self.valid_samples)} valid samples")
    
    def _process_sample(self, idx):
        sample = self.valid_samples[idx]
        enhanced_text = self._create_rich_description(sample['class_name'])
        
        result = self.audio_processor.load_and_preprocess(sample['audio_path'])
        if len(result) == 4:
            mel_spec, length, success, waveform = result
        else:
            mel_spec, length, success = result
        
        return {
            'audio_features': mel_spec,
            'text': enhanced_text,
            'audio_length': length,
            'domain': sample['domain'],
            'language': sample['language'],
            'dataset': sample['dataset'],
            'modality': sample['modality'],
            'success': success,
            'audio_path': sample['audio_path']
        }
    
    def _create_rich_description(self, class_name):
        descriptions = {
            'air_conditioner': 'An air conditioner running continuously',
            'car_horn': 'A car horn honking in an urban environment',
            'children_playing': 'Children laughing and playing outdoors',
            'dog_bark': 'A dog barking in the neighborhood',
            'drilling': 'Power drilling in a construction site'
        }
        name = f'The sound of {class_name.replace("_", " ")}'
        return f"Urban sound: {descriptions.get(class_name, name)}"

class SongDescriberDataset(BaseAudioDataset):
    """Song Describer dataset with robust MP3 handling"""
    
    def __init__(self, csv_path, audio_root, audio_processor, tokenizer, max_samples=None):
        super().__init__(audio_processor, tokenizer, max_samples)
        self.csv_path = csv_path
        self.audio_root = Path(audio_root)
        self._load_and_validate()
        
    def _load_and_validate(self):
        if not Path(self.csv_path).exists():
            raise FileNotFoundError(f"SongDescriber CSV not found: {self.csv_path}")
        metadata = pd.read_csv(self.csv_path)
        for idx, row in tqdm(metadata.iterrows(), total=len(metadata), desc="Validating SongDescriber"):
            audio_filename = None
            for col in ['audio_file', 'filename', 'file', 'audio_path', 'path', 'ytid']:
                if col in row and pd.notna(row[col]):
                    audio_filename = str(row[col])
                    break
            
            if not audio_filename:
                continue
            
            audio_path = self._find_audio_file(audio_filename)
            if audio_path is None:
                continue
            
            is_valid, info = self.audio_processor.validate_audio(audio_path)
            if not is_valid:
                continue
            
            description = self._get_description(row)
            if not description or len(description) < 10:
                continue
            
            self.valid_samples.append({
                'audio_path': str(audio_path),
                'description': description,
                'genre': str(row.get('genre', 'unknown')).lower(),
                'dataset': 'songdescriber',
                'domain': 'music',
                'language': 'en',
                'modality': 'music',
                'duration': info.get('duration', 0) if isinstance(info, dict) else 0
            })
            count += 1
        
        if not self.valid_samples:
            raise ValueError("No valid SongDescriber samples found")
        
        print(f"SongDescriber: {len(self.valid_samples)} valid samples")
    
    def _find_audio_file(self, audio_filename):
        """Find audio file with multiple path attempts"""
        base_name = Path(audio_filename).stem
        folder_num = base_name[:2] if len(base_name) >= 2 and base_name[:2].isdigit() else None
        extensions = ['.mp3', '.wav', '.flac', '.m4a', '.ogg']
        
        if folder_num:
            base_paths = [
                self.audio_root / 'audio_song_desc' / 'data' / 'audio' / 'audio' / folder_num,
            ]
        else:
            base_paths = [self.audio_root, self.audio_root / 'audio']
        
        for base_path in base_paths:
            for ext in extensions:
                candidate = base_path / audio_filename
                if candidate.exists():
                    return candidate
                candidate = base_path / f"{base_name}{ext}"
                if candidate.exists():
                    return candidate
        
        return None
    
    def _get_description(self, row):
        """Extract description from metadata row"""
        for col in ['caption', 'description', 'text', 'summary']:
            if col in row and pd.notna(row[col]):
                desc = str(row[col]).strip()
                if len(desc) > 0:
                    return desc
        return None
    
    def _process_sample(self, idx):
        sample = self.valid_samples[idx]
        enhanced_text = f"Music: {sample['description']}"
        
        result = self.audio_processor.load_and_preprocess(sample['audio_path'])
        if len(result) == 4:
            mel_spec, length, success, waveform = result
        else:
            mel_spec, length, success = result
        
        return {
            'audio_features': mel_spec,
            'text': enhanced_text,
            'audio_length': length,
            'domain': sample['domain'],
            'language': sample['language'],
            'dataset': sample['dataset'],
            'modality': sample['modality'],
            'success': success,
            'audio_path': sample['audio_path']
        }

# =============================================
# BALANCED SAMPLING DATASET - CORRECTED
# =============================================

class BalancedSamplingDataset(Dataset):
    """Simplified balanced sampling across all datasets"""
    
    def __init__(self, dataset_configs, audio_processor, tokenizer, samples_per_group=None, enforce_balance=True):
        self.audio_processor = audio_processor
        self.tokenizer = tokenizer
        self.enforce_balance = enforce_balance
        
        # Initialize all datasets
        self.datasets = self._initialize_datasets(dataset_configs)
        
        # Create a simple merged list of all samples
        self.all_samples = []
        self.dataset_indices = {}
        self._merge_all_datasets()
        
        # For balanced sampling
        self.dataset_names = list(self.datasets.keys())
        self.samples_per_dataset = {}
        for name in self.dataset_names:
            self.samples_per_dataset[name] = len(self.dataset_indices[name])
        
        self._print_summary()
    
    def _initialize_datasets(self, configs):
        """Initialize all configured datasets"""
        datasets = {}
        
        dataset_classes = {
            'ljspeech': LJSpeechDataset,
            'fma': FMADataset,
            'common_voice': CommonVoiceDataset,
            'esc50': ESC50Dataset,
            'urbansound8k': UrbanSound8KDataset,
            'songdescriber': SongDescriberDataset
        }
        
        for dataset_name in ['ljspeech', 'fma', 'common_voice', 'esc50', 'urbansound8k', 'songdescriber']:
            if dataset_name in configs:
                dataset_class = dataset_classes[dataset_name]
                config = configs[dataset_name]
                
                try:
                    if dataset_name in ['ljspeech', 'fma', 'common_voice']:
                        init_params = {
                            'root_dir': config['root_dir'],
                            'audio_processor': self.audio_processor,
                            'tokenizer': self.tokenizer,
                            'max_samples': config.get('max_samples')
                        }
                        
                        if dataset_name == 'common_voice':
                            if 'target_languages' in config:
                                init_params['target_languages'] = config['target_languages']
                        elif dataset_name == 'fma':
                            if 'subset' in config:
                                init_params['subset'] = config['subset']
                        
                        datasets[dataset_name] = dataset_class(**init_params)
                    else:
                        datasets[dataset_name] = dataset_class(
                            csv_path=config['csv_path'],
                            audio_root=config['audio_root'],
                            audio_processor=self.audio_processor,
                            tokenizer=self.tokenizer,
                            max_samples=config.get('max_samples')
                        )
                    
                    print(f"✓ {dataset_name} loaded: {len(datasets[dataset_name])} samples")
                except Exception as e:
                    print(f"✗ Failed to load {dataset_name}: {e}")
        
        if not datasets:
            raise ValueError("No datasets loaded successfully")
        
        return datasets
    
    def _merge_all_datasets(self):
        """Simply merge all datasets into a single list"""
        for dataset_name, dataset in self.datasets.items():
            self.dataset_indices[dataset_name] = []
            
            for local_idx in range(len(dataset)):
                global_idx = len(self.all_samples)
                self.all_samples.append({
                    'dataset_name': dataset_name,
                    'local_idx': local_idx
                })
                self.dataset_indices[dataset_name].append(global_idx)
    
    def _print_summary(self):
        """Print dataset summary"""
        print(f"\nBalanced Sampling Dataset Summary:")
        print(f"=" * 50)
        total_samples = 0
        for dataset_name, indices in self.dataset_indices.items():
            count = len(indices)
            total_samples += count
            print(f"{dataset_name}: {count} samples")
        print(f"-" * 50)
        print(f"Total samples: {total_samples}")
        print(f"Enforce balance: {self.enforce_balance}")
        print(f"=" * 50)
    
    def __len__(self):
        """Return appropriate length based on sampling strategy"""
        if self.enforce_balance:
            # For balanced sampling, length is max dataset size * num datasets
            max_size = max(self.samples_per_dataset.values()) if self.samples_per_dataset else 0
            return max_size * len(self.dataset_names)
        else:
            # For simple concatenation
            return len(self.all_samples)
    
    def __getitem__(self, idx):
        """Get sample with optional balanced sampling"""
        if self.enforce_balance:
            # Round-robin through datasets
            dataset_idx = idx % len(self.dataset_names)
            dataset_name = self.dataset_names[dataset_idx]
            
            # Get sample index within the dataset
            dataset_size = self.samples_per_dataset[dataset_name]
            if dataset_size == 0:
                # Fallback to first non-empty dataset
                for name in self.dataset_names:
                    if self.samples_per_dataset[name] > 0:
                        dataset_name = name
                        dataset_size = self.samples_per_dataset[name]
                        break
            
            within_dataset_idx = (idx // len(self.dataset_names)) % dataset_size
            
            # Get the actual sample
            return self.datasets[dataset_name][within_dataset_idx]
        else:
            # Simple sequential access
            if idx >= len(self.all_samples):
                idx = idx % len(self.all_samples)
            
            sample_info = self.all_samples[idx]
            dataset_name = sample_info['dataset_name']
            local_idx = sample_info['local_idx']
            
            return self.datasets[dataset_name][local_idx]
    
    def get_batch_statistics(self, batch_size=32):
        """Get statistics about a typical batch"""
        if self.enforce_balance:
            samples_per_dataset = batch_size // len(self.dataset_names)
            remainder = batch_size % len(self.dataset_names)
            
            stats = {}
            for i, name in enumerate(self.dataset_names):
                count = samples_per_dataset + (1 if i < remainder else 0)
                stats[name] = count
            
            return {
                'batch_size': batch_size,
                'distribution': stats,
                'balanced': True
            }
        else:
            return {
                'batch_size': batch_size,
                'distribution': 'Sequential from merged datasets',
                'balanced': False
            }