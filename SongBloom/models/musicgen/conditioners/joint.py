from pathlib import Path
import re
from .base import *
import torchaudio
import torchaudio.transforms as T


class MuLANConditioner(JointEmbeddingConditioner):
    """ MERT based feature extractor
    """
    def __init__(self, model_dim: int, output_dim: int, sample_rate: int, 
                 config_path: str, 
                 quantize: bool = False, n_q: int=12, n_bin: int=1024,
                 discard_text: bool = False, t2a_adaptor: dict = None, **kwargs):
                
        # import pdb; pdb.set_trace()
        super().__init__(dim=model_dim, output_dim=output_dim)
        # import pdb; pdb.set_trace()
        from ...mulan import get_mulan
        self.mulan_embedder = get_mulan(config_path)
        for param in self.mulan_embedder.parameters():
            param.requires_grad = False

        
        self.sample_rate = sample_rate
        self.resampler = T.Resample(self.sample_rate, self.mulan_embedder.sr)
        self.mulan_embedder.eval()

        # NOTE by YCY, 去掉不需要的层数，减少运算时间/显存
        if self.mulan_embedder.mulan.audio.use_layer_idx != -1:
            self.mulan_embedder.mulan.audio.model.encoder.layers = \
                self.mulan_embedder.mulan.audio.model.encoder.layers[:self.mulan_embedder.mulan.audio.use_layer_idx + 1] 
        
        self.quantizer = None
        if quantize:
            from quantization import ResidualVectorQuantizer
            self.quantizer = ResidualVectorQuantizer(model_dim, n_q=n_q, bins=n_bin)
            raise NotImplementedError
        # delattr(self.mulan_embedder.mulan, "contrast")
        # delattr(self.mulan_embedder.mulan.audio, "aggregator")
        if discard_text:
            delattr(self.mulan_embedder.mulan, "text")
            delattr(self.mulan_embedder.mulan, "text_to_latents")
        
 
        self.t2a_adaptor = None
        if t2a_adaptor is not None:
            import os, sys
            sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
            from t2aembd import T2AemdAdaptor
            self.t2a_adaptor = T2AemdAdaptor(**t2a_adaptor)


    def _get_wav_embedding(self, wav: torch.Tensor) -> torch.Tensor:
        """Get the wav embedding from the WavCondition.
        The conditioner will either extract the embedding on-the-fly computing it from the condition wav directly
        or will rely on the embedding cache to load the pre-computed embedding if relevant.
        """
        bsz = wav.shape[0]
        # import pdb; pdb.set_trace()
        # self.resampler = self.resampler.cuda()
        is_droped = (wav.shape[-1] == 1)
        if is_droped: # droped
            return torch.zeros((bsz, 1, self.mulan_embedder.mulan.dim_latent), device=wav.device, dtype=wav.dtype)

        input_audio = self.resampler(wav.squeeze(1))

        if input_audio.ndim == 3 and input_audio.shape[1] == 1:
            input_audio = input_audio.squeeze(1)

        # B, T
        self.mulan_embedder = self.mulan_embedder.to(torch.float32)
        with torch.cuda.amp.autocast(enabled=False):
            # Inference Check
            audio_embeds = self.mulan_embedder(wavs = input_audio.float())
            # audio_embeds = self.mulan_embedder(wavs = input_audio.float()) # [B, T] -> [B, 512]
        audio_embeds = audio_embeds.to(input_audio.dtype)
        audio_embeds = audio_embeds.unsqueeze(1)    

        return audio_embeds


    def _get_text_embedding(self, texts) -> torch.Tensor:
        """Get the wav embedding from the WavCondition.
        The conditioner will either extract the embedding on-the-fly computing it from the condition wav directly
        or will rely on the embedding cache to load the pre-computed embedding if relevant.
        """
        # with torch.cuda.amp.autocast(enabled=False):
        dropped = torch.tensor([(i is None) for i in texts], device=self.mulan_embedder.device)
        texts = [i if i is not None else "" for i in texts]
        # print(texts)
        self.mulan_embedder = self.mulan_embedder.to(torch.float32)
        with torch.cuda.amp.autocast(enabled=False):
            if self.t2a_adaptor is None:
                text_embeds = self.mulan_embedder(texts = texts)
            else:
                text_seq_embeds, mask, text_embeds = self.mulan_embedder(texts = texts, return_frames=True)
                syns_aembs = []
                for _seq, _mask, _temb in zip(text_seq_embeds, mask, text_embeds):
                    syns_aembs.append(self.t2a_adaptor(_temb.unsqueeze(0), _seq.unsqueeze(0), _mask.unsqueeze(0)))
                text_embeds = torch.cat(syns_aembs, dim=0)
                
        text_embeds = torch.masked_fill(text_embeds, dropped.unsqueeze(-1), 0)
        text_embeds = text_embeds.unsqueeze(1)
        # import pdb; pdb.set_trace()
        return text_embeds

    def tokenize(self, x: JointEmbedCondition) -> JointEmbedCondition:
        """Apply WavConditioner tokenization and populate cache if needed."""
        return x

    def forward(self, x: JointEmbedCondition) -> ConditionType:
        """Extract condition embedding and mask from a waveform and its metadata.
        Args:
            x (WavCondition): Waveform condition containing raw waveform and metadata.
        Returns:
            ConditionType: a dense vector representing the conditioning along with its mask
        """
        
        # import pdb; pdb.set_trace()
        if any([i is not None for i in x.text]):
            assert x.wav.shape[-1] == 1 # wav must be None
            lengths = torch.ones(len(x.text)).to(self.output_proj.weight)
            embeds = self._get_text_embedding(x.text)
        else:
            # import pdb; pdb.set_trace()
            embeds = self._get_wav_embedding(x.wav)
            lengths = x.length
            # import pdb; pdb.set_trace()
        # import pdb; pdb.set_trace()
        embeds = embeds.to(self.output_proj.weight)
        embeds = self.output_proj(embeds)
        lengths = torch.ones_like(lengths)
        mask = length_to_mask(lengths, max_len=embeds.shape[1]).int()  # type: ignore
        embeds = (embeds * mask.unsqueeze(2).to(self.output_proj.weight))

        return embeds, mask



