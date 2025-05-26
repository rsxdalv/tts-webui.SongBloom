from .base import *

import spacy
import warnings
import random
import hashlib
from transformers import RobertaTokenizer, T5EncoderModel, T5Tokenizer, AutoTokenizer, XLMRobertaModel, XLMRobertaTokenizer  # type: ignore
from num2words import num2words

def hash_trick(word: str, vocab_size: int) -> int:
    """Hash trick to pair each word with an index

    Args:
        word (str): word we wish to convert to an index
        vocab_size (int): size of the vocabulary
    Returns:
        int: index of the word in the embedding LUT
    """

    hash = int(hashlib.sha256(word.encode("utf-8")).hexdigest(), 16)
    return hash % vocab_size


class Tokenizer:
    """Base tokenizer implementation
    (in case we want to introduce more advances tokenizers in the future).
    """
    def __call__(self, texts: tp.List[tp.Optional[str]]) -> tp.Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError()


class WhiteSpaceTokenizer(Tokenizer):
    """This tokenizer should be used for natural language descriptions.
    For example:
    ["he didn't, know he's going home.", 'shorter sentence'] =>
    [[78, 62, 31,  4, 78, 25, 19, 34],
    [59, 77,  0,  0,  0,  0,  0,  0]]
    """
    PUNCTUATION = "?:!.,;"

    def __init__(self, n_bins: int, pad_idx: int = 0, language: str = "en_core_web_sm",
                 lemma: bool = True, stopwords: bool = True) -> None:
        self.n_bins = n_bins
        self.pad_idx = pad_idx
        self.lemma = lemma
        self.stopwords = stopwords
        try:
            self.nlp = spacy.load(language)
        except IOError:
            spacy.cli.download(language)  # type: ignore
            self.nlp = spacy.load(language)

    @tp.no_type_check
    def __call__(self, texts: tp.List[tp.Optional[str]],
                 return_text: bool = False) -> tp.Tuple[torch.Tensor, torch.Tensor]:
        """Take a list of strings and convert them to a tensor of indices.

        Args:
            texts (list[str]): List of strings.
            return_text (bool, optional): Whether to return text as additional tuple item. Defaults to False.
        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                - Indices of words in the LUT.
                - And a mask indicating where the padding tokens are
        """
        output, lengths = [], []
        texts = deepcopy(texts)
        for i, text in enumerate(texts):
            # if current sample doesn't have a certain attribute, replace with pad token
            if text is None:
                output.append(torch.Tensor([self.pad_idx]))
                lengths.append(0)
                continue

            # convert numbers to words
            text = re.sub(r"(\d+)", lambda x: num2words(int(x.group(0))), text)  # type: ignore
            # normalize text
            text = self.nlp(text)  # type: ignore
            # remove stopwords
            if self.stopwords:
                text = [w for w in text if not w.is_stop]  # type: ignore
            # remove punctuation
            text = [w for w in text if w.text not in self.PUNCTUATION]  # type: ignore
            # lemmatize if needed
            text = [getattr(t, "lemma_" if self.lemma else "text") for t in text]  # type: ignore

            texts[i] = " ".join(text)
            lengths.append(len(text))
            # convert to tensor
            tokens = torch.Tensor([hash_trick(w, self.n_bins) for w in text])
            output.append(tokens)

        mask = length_to_mask(torch.IntTensor(lengths)).int()
        padded_output = pad_sequence(output, padding_value=self.pad_idx).int().t()
        if return_text:
            return padded_output, mask, texts  # type: ignore
        return padded_output, mask


class NoopTokenizer(Tokenizer):
    """This tokenizer should be used for global conditioners such as: artist, genre, key, etc.
    The difference between this and WhiteSpaceTokenizer is that NoopTokenizer does not split
    strings, so "Jeff Buckley" will get it's own index. Whereas WhiteSpaceTokenizer will
    split it to ["Jeff", "Buckley"] and return an index per word.

    For example:
    ["Queen", "ABBA", "Jeff Buckley"] => [43, 55, 101]
    ["Metal", "Rock", "Classical"] => [0, 223, 51]
    """
    def __init__(self, n_bins: int, pad_idx: int = 0):
        self.n_bins = n_bins
        self.pad_idx = pad_idx

    def __call__(self, texts: tp.List[tp.Optional[str]]) -> tp.Tuple[torch.Tensor, torch.Tensor]:
        output, lengths = [], []
        for text in texts:
            # if current sample doesn't have a certain attribute, replace with pad token
            if text is None:
                output.append(self.pad_idx)
                lengths.append(0)
            else:
                output.append(hash_trick(text, self.n_bins))
                lengths.append(1)

        tokens = torch.LongTensor(output).unsqueeze(1)
        mask = length_to_mask(torch.IntTensor(lengths)).int()
        return tokens, mask




# Hard code T5 tokenizer
class T5Conditioner(TextConditioner):
    """T5-based TextConditioner.

    Args:
        name (str): Name of the T5 model.
        output_dim (int): Output dim of the conditioner.
        finetune (bool): Whether to fine-tune T5 at train time.
        autocast_dtype (tp.Optional[str], optional): Autocast dtype.
        word_dropout (float, optional): Word dropout probability.
        normalize_text (bool, optional): Whether to apply text normalization.
    """
    MODELS = ["t5-small", "t5-base", "t5-large", "t5-3b", "t5-11b",
              "google/flan-t5-small", "google/flan-t5-base", "google/flan-t5-large",
              "google/flan-t5-xl", "google/flan-t5-xxl", "xlm-roberta-base"]
    MODELS_DIMS = {
        "t5-small": 512,
        "t5-base": 768,
        "t5-large": 1024,
        "t5-3b": 1024,
        "t5-11b": 1024,
        "xlm-roberta-base": 768,
        "google/flan-t5-small": 512,
        "google/flan-t5-base": 768,
        "google/flan-t5-large": 1024,
        "google/flan-t5-3b": 1024,
        "google/flan-t5-11b": 1024,
    }

    def __init__(self, name: str, output_dim: int, finetune: bool = 'False', 
                 autocast_dtype: tp.Optional[str] = 'float32', word_dropout: float = 0.,
                 normalize_text: bool = False):
        assert name in self.MODELS, f"Unrecognized t5 model name (should in {self.MODELS})"
        super().__init__(self.MODELS_DIMS[name], output_dim)
        self.name = name
        self.word_dropout = word_dropout
        # Let's disable logging temporarily because T5 will vomit some errors otherwise.
        # thanks https://gist.github.com/simon-weber/7853144
        previous_level = logging.root.manager.disable
        logging.disable(logging.ERROR)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                if 'roberta' in name:
                    self.t5_tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-base", local_files_only=True)
                    t5 = XLMRobertaModel.from_pretrained("xlm-roberta-base", local_files_only=True).to('cpu')
                else:
                    self.t5_tokenizer = T5Tokenizer.from_pretrained(name, local_files_only=True)
                    t5 = T5EncoderModel.from_pretrained(name, local_files_only=True).train(mode=finetune)
            finally:
                logging.disable(previous_level)

        for param in t5.parameters():
            param.requires_grad = False

        self.t5 = t5 
        # if finetune:
        #     self.t5 = t5
        # else:
        #     # this makes sure that the t5 models is not part
        #     # of the saved checkpoint
        #     self.__dict__['t5'] = t5

        self.normalize_text = normalize_text
        if normalize_text:
            self.text_normalizer = WhiteSpaceTokenizer(1, lemma=True, stopwords=True)

    def tokenize(self, x: tp.List[tp.Optional[str]]) -> tp.Dict[str, torch.Tensor]:
        
        # import pdb; pdb.set_trace()
        # if current sample doesn't have a certain attribute, replace with empty string

        entries: tp.List[str] = [xi if xi is not None else "" for xi in x]
        if self.normalize_text:
            _, _, entries = self.text_normalizer(entries, return_text=True)
        if self.word_dropout > 0. and self.training:
        # if self.word_dropout > 0. :
            new_entries = []
            for entry in entries:
                words = [word for word in entry.split(" ") if random.random() >= self.word_dropout]
                new_entries.append(" ".join(words))
            entries = new_entries

        empty_idx = torch.LongTensor([i for i, xi in enumerate(entries) if xi == ""])
        inputs = self.t5_tokenizer(entries, return_tensors='pt', padding=True)
        mask = inputs['attention_mask']
        mask[empty_idx, :] = 0  # zero-out index where the input is non-existant
        return inputs

    def forward(self, inputs: tp.Dict[str, torch.Tensor]) -> ConditionType:
        # import pdb; pdb.set_trace()
        mask = inputs['attention_mask']
        inputs['input_ids'] =  inputs['input_ids'].to(self.output_proj.weight.device)
        inputs['attention_mask'] =  inputs['attention_mask'].to(self.output_proj.weight.device)
        with torch.set_grad_enabled(False):
            with torch.autocast(enabled=True, device_type='cuda', dtype=torch.float16):
                embeds = self.t5(**inputs).last_hidden_state
        
        embeds = self.output_proj(embeds.to(self.output_proj.weight))
        mask = mask.to(self.output_proj.weight)

        embeds = (embeds * mask.unsqueeze(-1))
        return embeds, mask



class LlamaTokenizerConditioner(TextConditioner):
    def __init__(self, output_dim: int, max_len = 300, padding_idx='</s>', tokenizer_type=None,
                 pretrained="hf-internal-testing/llama-tokenizer", add_token_list=[], type_tokens=[]): #"hfl/chinese-llama-2-13b"
        from transformers import LlamaTokenizer
        print(f"text tokenizer from {pretrained}")
        self.text_tokenizer = LlamaTokenizer.from_pretrained(pretrained) 
        if add_token_list != []:
            self.text_tokenizer.add_tokens(add_token_list, special_tokens=True)
        # voc_size = self.text_tokenizer.vocab_size
        voc_size = len(self.text_tokenizer.get_vocab()) # 加了额外token之后vocab_size似乎不会额外增加 ——cyy
        # print(self.text_tokenizer.get_added_vocab(), voc_size)
        # import pdb; pdb.set_trace()
        padding_idx = str(padding_idx)
        super().__init__(voc_size, output_dim, True, 2)
        
        self.text_tokenizer.pad_token = padding_idx
        self.max_len = max_len
        self.padding_idx = padding_idx

        vocab = self.text_tokenizer.get_vocab()
        self.type_token_ids = [vocab[i] for i in type_tokens if i in vocab]
        struct_tokens = [padding_idx] + [i for i in add_token_list if i[0]=='[' and i[-1]==']']
        self.struct_token_ids = [vocab[i] for i in struct_tokens]
        print("type tokens: ",{self.text_tokenizer.convert_ids_to_tokens(i):i for i in self.type_token_ids},
                 "\t all structure tokens: ", {self.text_tokenizer.convert_ids_to_tokens(i):i for i in self.struct_token_ids})
        
    def tokenize(self, x: tp.List[tp.Optional[str]]) -> tp.Dict[str, torch.Tensor]:
        x = [xi if xi is not None else "" for xi in x]
        inputs = self.text_tokenizer(x, return_tensors="pt", padding=True)
        # print(x, [self.text_tokenizer.convert_ids_to_tokens(i.tolist()) for i in inputs['input_ids']])
        # import pdb; pdb.set_trace()
        # print(x, inputs['input_ids'].shape)
        return inputs

    def forward(self, inputs: tp.Dict[str, torch.Tensor]) -> ConditionType:
        '''
        自动将type embedding 从拼接变成sum
        没有的话也不影响
        '''
        mask = inputs['attention_mask']
        tokens = inputs['input_ids']
        B = tokens.shape[0]

        is_tp_embed = torch.any(torch.stack([tokens == i for i in self.type_token_ids], dim=-1),dim=-1)
        is_sp_embed = torch.any(torch.stack([tokens == i for i in self.struct_token_ids], dim=-1),dim=-1)

        tp_cover_range = torch.zeros_like(tokens)
        for b, (is_tp, is_sp) in enumerate(zip(is_tp_embed, is_sp_embed)):
            tp_list = torch.where(is_tp)[0].tolist()
            sp_list = torch.where(is_sp)[0].tolist()
            sp_list.append(mask[b].sum())
            n = 1
            for i, st in enumerate(sp_list[:-1]):
                if st in tp_list:
                    tp_cover_range[b, st: sp_list[i+1]] = n
                    n += 1

        type_tokens = pad_sequence([torch.masked_select(tokens[b], is_tp_embed[b]) for b in range(B)], batch_first=True, padding_value=2)
        tokens = pad_sequence([torch.masked_select(tokens[b], ~is_tp_embed[b]) for b in range(B)], batch_first=True, padding_value=2)
        mask = pad_sequence([torch.masked_select(mask[b], ~is_tp_embed[b]) for b in range(B)], batch_first=True, padding_value=0)
        tp_cover_range = pad_sequence([torch.masked_select(tp_cover_range[b], ~is_tp_embed[b]) for b in range(B)], batch_first=True, padding_value=0)

        if self.max_len is not None:
            if inputs['input_ids'].shape[-1] > self.max_len:
                warnings.warn(f"Max len limit ({self.max_len}) Exceed! \
                              {[self.text_tokenizer.convert_ids_to_tokens(i.tolist()) for i in tokens]} will be cut!")
            tokens = self.pad_2d_tensor(tokens, self.max_len, 2).to(self.output_proj.weight.device)
            mask = self.pad_2d_tensor(mask, self.max_len, 0).to(self.output_proj.weight.device)
            tp_cover_range = self.pad_2d_tensor(tp_cover_range, self.max_len, 0).to(self.output_proj.weight.device)
    
        embeds = self.output_proj(tokens)
        type_embeds = self.output_proj(type_tokens.to(self.output_proj.weight.device))
        type_embeds = F.pad(type_embeds, (0, 0, 1, 0))
        gathered_type_embeds = torch.gather(dim=1, index=tp_cover_range.unsqueeze(-1).expand(-1,-1,type_embeds.shape[-1]), input=type_embeds)
        embeds += gathered_type_embeds
        return embeds, mask
    
    def pad_2d_tensor(self, x, max_len, pad_id):
        # 获取输入 tensor 的形状
        batch_size, seq_len = x.size()
        # 计算需要填充的长度
        pad_len = max_len - seq_len

        # 如果需要填充
        if pad_len > 0:
            # 创建填充 tensor
            pad_tensor = torch.full((batch_size, pad_len), pad_id, dtype=x.dtype, device=x.device)

            # 沿第二个维度（列）连接输入 tensor 和填充 tensor
            padded_tensor = torch.cat([x, pad_tensor], dim=1)
        elif pad_len < 0:
            padded_tensor = x[:, :max_len]
        else:
            # 如果不需要填充，直接返回输入 tensor
            padded_tensor = x

        return padded_tensor



class PhonemeTokenizerConditioner(TextConditioner):
    def __init__(self, 
                 output_dim: int, 
                 vocab_list,
                 max_len = 600, 
                 max_sentence_per_structure = 50,
                 structure_tokens=None,
                 structure_split_tokens=[','],
                 sentence_split_tokens=['.'],
                 mode='sum',
                 structure_output_dim = 64,
                 sentence_output_dim = 64,
                 max_duration = 120,
                 interpolate = False,
                 ): 
        
        self.vocab_list = vocab_list
        self.max_len = max_len
        self.mode = mode
        self.max_sentence_per_structure = max_sentence_per_structure
        voc_size = len(self.vocab_list)
        self.interpolate = interpolate
        
        if structure_tokens is None:
            structure_tokens = [i for i in vocab_list if len(i) > 1 and i[0] == '[' and i[-1] == ']']
        self.structure_token_ids = [vocab_list.index(i) for i in structure_tokens if i in vocab_list]
        self.structure_split_token_ids = [vocab_list.index(i) for i in structure_split_tokens]
        self.sentence_split_token_ids = [vocab_list.index(i) for i in sentence_split_tokens]

        # here initialize a output_proj (nn.Embedding) layer
        # By default the first vocab is "" (null)
        if mode == 'sum':
            content_output_dim = output_dim
            sentence_output_dim = output_dim
            structure_output_dim = output_dim
        else:   # concat
            content_output_dim = output_dim - sentence_output_dim - structure_output_dim   # by default
            
        super().__init__(voc_size, content_output_dim, input_token=True, padding_idx=0)
        if self.mode != 'sum':
            self.special_emb = nn.Embedding(len(self.structure_token_ids)+len(self.structure_split_token_ids)+len(self.sentence_split_token_ids)+1, 
                                             structure_output_dim, padding_idx=0)
            
        self.blank_emb = nn.Parameter(torch.zeros(1, output_dim), requires_grad=False)

        # the first index is "empty structure" token
        self.sentence_idx_in_structure_emb = nn.Embedding(max_sentence_per_structure, sentence_output_dim, padding_idx=0) 

        # print("max_len", self.max_len)
        print(self.structure_token_ids)
        
        self.resolution = max_duration / max_len    # e.g., 120 / 600 = 0.2s 
        print(self.__class__, f"resolution = {self.resolution}")
    
    def tokenize(self, x: tp.List[tp.Optional[str]]) -> tp.Dict[str, torch.Tensor]:
        inputs = []
        for xx in x:
            xx = '' if xx is None else xx
            vocab_id = [self.vocab_list.index(item) for item in xx.split(" ") if item in self.vocab_list]
            inputs.append(torch.tensor(vocab_id).long()) # [T]        
        return inputs
    
    
    def interpolate_with_structure_duration(self, special_tokens, embeds, structure_dur):
        # embeds: [T, N]
        def sec2idx(sec):   # convert duration sec to token index
            return int(sec / self.resolution)
        
        def target_token_types2list(tokens, target_token_types):

            is_target_list = torch.any(torch.stack([tokens == i for i in target_token_types], dim=-1), dim=-1)
            is_target_list = torch.where(is_target_list)[0].tolist()
            return is_target_list
        
        structure_ids = []
        for (structure, st, et) in structure_dur:
            structure_ids.append([structure, sec2idx(st), sec2idx(et)])
            
        """
        interpolate embeddings of each structure according to its duration 
        """
        is_structure_list = target_token_types2list(special_tokens, self.structure_token_ids)
        is_structure_list.append(special_tokens.shape[-1])
        
        split_tokens = deepcopy(self.structure_split_token_ids)
        split_tokens.extend(self.sentence_split_token_ids)
        # is_split_list = target_token_types2list(special_tokens, split_tokens)
                
        
        interpolated_embeds = embeds[:is_structure_list[0]]
        for i, st in enumerate(is_structure_list[:-1]):
            # (lorry) Explain "-tmp": 
            # All structures are connected with " , " token,
            # " ," is also the final token of each structure except the final one,
            # but here we dont want to interpolate " , " token
            tmp = 1
            if i == len(is_structure_list[:-1]) - 1:  # the final structure, no need for "-1"
                tmp = 0
            
       #     print(st, is_structure_list[i+1]-tmp)
            to_interpolate = embeds[st: is_structure_list[i+1] - tmp]
            interpolate_size = structure_ids[i][2] - structure_ids[i][1] - tmp
       #     print(interpolate_size)
            
            #import pdb; pdb.set_trace()
            # print(interpolated_embeds.shape, to_interpolate.shape, interpolate_size, )
            if to_interpolate.shape[0] == 0:
                import pdb; pdb.set_trace()
            this_interpolated_embeds = F.interpolate(to_interpolate.unsqueeze(0).transpose(2, 1), 
                                        size=interpolate_size, 
                                        mode='nearest-exact').squeeze(0).transpose(1, 0)
            
            if tmp == 1:
                interpolated_embeds = torch.cat((interpolated_embeds, this_interpolated_embeds, 
                                                embeds[is_structure_list[i+1]].unsqueeze(0)), 0)
            else:
                interpolated_embeds = torch.cat((interpolated_embeds, this_interpolated_embeds), 0)
        return interpolated_embeds
            
            
    def forward(self, batch_tokens: tp.List, structure_dur = None) -> ConditionType:
        """
        Encode token_id into three types of embeddings:
        1) content embedding: phoneme only (or meaningful contents to be sung out) 
        2) structure embedding: structure / separation embeddings, including structures (verse/chorus/...), separators (. / ,)
        The two above share the same embedding layer, can be changed to separate embedding layers.
        3) sentence_idx embedding (per structure): 
        """
        embeds_batch = []
        # print(batch_tokens)
        for b in range(len(batch_tokens)):
            tokens = batch_tokens[b]  

            content_tokens = torch.zeros_like(tokens)
            special_tokens = torch.zeros_like(tokens)
            sentence_idx_in_structure_tokens = torch.zeros_like(tokens) 

            current_structure_idx = 1
            current_sentence_in_structure_idx = 1
            current_structure = 0

            for i in range(tokens.shape[-1]):
                token = tokens[i]
                if token in self.structure_token_ids:       # structure token
                    # only update structure token, leave content and sentence index token null (default 0)
                    if self.mode == 'sum':
                        special_tokens[i] = token
                    else:
                        special_tokens[i] = self.structure_token_ids.index(token) + 1
                    current_structure = token
                    current_structure_idx += 1
                    current_sentence_in_structure_idx = 1

                elif token in self.sentence_split_token_ids:    # utterance split token
                    # only update structure token, leave content and sentence index token null (default 0)
                    # add up sentence index
                    if self.mode == 'sum':
                        special_tokens[i] = token
                    else:
                        special_tokens[i] = self.sentence_split_token_ids.index(token) + 1 + len(self.structure_token_ids)
                    current_sentence_in_structure_idx += 1

                elif token in self.structure_split_token_ids:    # structure split token
                    # update structure token (current structure), content token (current token), 
                    # blank index token 
                    if self.mode == 'sum':
                        special_tokens[i] = token
                    else:
                        special_tokens[i] = self.structure_split_token_ids.index(token) + 1 + len(self.structure_token_ids) + len(self.sentence_split_token_ids)

                else:       # content tokens
                    content_tokens[i] = token
                    special_tokens[i] = current_structure
                    sentence_idx_in_structure_tokens[i] = min(current_sentence_in_structure_idx, self.max_sentence_per_structure - 1)

            # print("tokens", tokens.max(), tokens.min())
            # print("special tokens", special_tokens.max(), special_tokens.min())
            # print("sentence idx in structure", sentence_idx_in_structure_tokens.max(), sentence_idx_in_structure_tokens.min())
            device = self.output_proj.weight.device
            
            # import pdb; pdb.set_trace()
            content_embeds = self.output_proj(tokens.to(device))    # [T, N]
            if self.mode == 'sum':
                structure_embeds = self.output_proj(special_tokens.to(device))
            else:
                structure_embeds = self.special_emb(special_tokens.to(device))
            sentence_idx_embeds = self.sentence_idx_in_structure_emb(sentence_idx_in_structure_tokens.to(device))

            if self.mode == 'sum':
                embeds = content_embeds + structure_embeds + sentence_idx_embeds
            else:
                embeds = torch.cat((content_embeds, structure_embeds, sentence_idx_embeds), -1) # [T, N]
                
            if self.interpolate:
                embeds = self.interpolate_with_structure_duration(tokens, embeds, structure_dur[b])
            embeds_batch.append(embeds)

        # set batch_size = 1, [B, T, N]
        if self.max_len is not None:
            max_len = self.max_len
        else:
            max_len = max([e.shape[0] for e in embeds_batch])
        embeds, mask = self.pad_2d_tensor(embeds_batch, max_len)
        
        return embeds, mask
    
    
    def pad_2d_tensor(self, xs, max_len):
        new_tensor = []
        new_mask = []
        for x in xs:
            seq_len, dim = x.size()
            pad_len = max_len - seq_len

            if pad_len > 0:
                pad_tensor = self.blank_emb.repeat(pad_len, 1).to(x.device)  # T, D
                padded_tensor = torch.cat([x, pad_tensor], dim=0)
                mask = torch.cat((torch.ones_like(x[:, 0]), 
                                  torch.zeros_like(pad_tensor[:, 0])), 0)   # T
            elif pad_len < 0:
                padded_tensor = x[:max_len]
                mask = torch.ones_like(padded_tensor[:, 0])
            else:
                padded_tensor = x
                mask = torch.ones_like(x[:, 0])

            new_tensor.append(padded_tensor)
            new_mask.append(mask)
        # [B, T, D] & [B, T]
        return torch.stack(new_tensor, 0), torch.stack(new_mask, 0)   
