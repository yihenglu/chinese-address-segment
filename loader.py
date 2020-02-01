import os
import re
import codecs

from data_utils import create_dico, create_mapping, zero_digits
from data_utils import iob2, iob_iobes, get_seg_features


def load_sentences(path, lower, zeros):
    """
    Load sentences. A line must contain at least a word and its tag.
    Sentences are separated by empty lines.
    """
    sentences = []
    sentence = []
    num = 0
    for line in open(path, 'r',encoding='utf8'):  #其中一行，如“长 B-Dur”
        # if num == 18053727 or num == 18053726:
        #     print(line)
        # t = line
        num+=1
        # line = zero_digits(line.rstrip()) if zeros else line.rstrip()  #若忽略大小写，则就处理，rstrip()表示删除 string 字符串末尾的指定字符（默认为空格）
        line = zero_digits(line.replace('　','')) if zeros else line.rstrip()  #若忽略大小写，则就处理，rstrip()表示删除 string 字符串末尾的指定字符（默认为空格）
        # print(list(line))
        if not line:  #Sentences are separated by empty lines. 若处理到空行，则保存之前处理过的句子
            if len(sentence) > 0:
                sentences.append(sentence)
                sentence = []
        else:
            if line[0] == " ":  #若出现“  O”这种情况
                line = "$" + line[1:]
                word = line.split()
                # word[0] = " "
            else:
                word= line.split( )
            # if len(word) != 2:
            #     print(sentence)
            #     print(word)
            #     print(num)
            #     print(t)
            assert len(word) == 2
            sentence.append(word)
    if len(sentence) > 0:  #保存最后一句话处理的结果
        if 'DOCSTART' not in sentence[0][0]:
            sentences.append(sentence)
    return sentences


def update_tag_scheme(sentences, tag_scheme):
    """
    Check and update sentences tagging scheme to IOB2.
    Only IOB1 and IOB2 schemes are accepted.  #为什么只有这两个体系可用？？？
    """
    for i, s in enumerate(sentences):
        tags = [w[-1] for w in s]
        # Check that tags are given in the IOB format
        if not iob2(tags):
            s_str = '\n'.join(' '.join(w) for w in s)
            raise Exception('Sentences should be given in IOB format! ' +
                            'Please check sentence %i:\n%s' % (i, s_str))
        #以下转标记体系没看懂？？？
        if tag_scheme == 'iob':
            # If format was IOB1, we convert to IOB2
            for word, new_tag in zip(s, tags):
                word[-1] = new_tag
        elif tag_scheme == 'iobes':
            new_tags = iob_iobes(tags)
            for word, new_tag in zip(s, new_tags):
                word[-1] = new_tag
        else:
            raise Exception('Unknown tagging scheme!')


def char_mapping(sentences, lower):  #lower表示是否忽略大小写
    """
    Create a dictionary and a mapping of words, sorted by frequency.
    """
    chars = [[x[0].lower() if lower else x[0] for x in s] for s in sentences]  #获取句子中的字
    dico = create_dico(chars)  #create_dico表示创建词频字典
    dico["<PAD>"] = 10000001  #应该不是表示结尾？？？
    dico['<UNK>'] = 10000000
    char_to_id, id_to_char = create_mapping(dico)
    #print("Found %i unique words (%i in total)" % (
    #    len(dico), sum(len(x) for x in chars)
    #))
    return dico, char_to_id, id_to_char


def tag_mapping(sentences):
    """
    Create a dictionary and a mapping of tags, sorted by frequency.
    """
    
    f=open('tag_to_id.txt','w',encoding='utf8')
    f1=open('id_to_tag.txt','w',encoding='utf8')
    tags=[]
    for s in sentences:
        ts=[]
        for char in s:
            tag=char[-1]
            ts.append(tag)
        tags.append(ts)
    
    #tags1 = [[char[-1] for char in s] for s in sentences]
    dico = create_dico(tags)
    tag_to_id, id_to_tag = create_mapping(dico)
    #print("Found %i unique named entity tags" % len(dico))
    for k,v in tag_to_id.items():
        f.write(k+":"+str(v)+"\n")
    for k,v in id_to_tag.items():
        f1.write(str(k) + ":" + str(v) + "\n")
    return dico, tag_to_id, id_to_tag


def prepare_dataset(sentences, char_to_id, tag_to_id, lower=False, train=True):
    
    """
    Prepare the dataset. Return a list of lists of dictionaries containing:
        - word indexes
        - word char indexes
        - tag indexes
    """

    none_index = tag_to_id["O"]

    def f(x):
        return x.lower() if lower else x
    data = []
    for s in sentences:
        #print(sentences)
        string = [w[0] for w in s]
        chars = [char_to_id[f(w) if f(w) in char_to_id else '<UNK>']
                 for w in string]
        segs = get_seg_features("".join(string))
        if train:
            tags = [tag_to_id[w[-1]] for w in s]
        else:
            tags = [none_index for _ in chars]
        data.append([string, chars, segs, tags])  # 字符、字符id、根据词的长度，标记为0、1 2、1 2 3、1 2 2 3等、tag的id

    return data


def augment_with_pretrained(dictionary, ext_emb_path, chars):
    """
    Augment（增大） the dictionary with words that have a pretrained embedding.
    If `words` is None, we add every word that has a pretrained embedding
    to the dictionary, otherwise, we only add the words that are given by
    `words` (typically the words in the development and test sets.)
    """
    #print('Loading pretrained embeddings from %s...' % ext_emb_path)
    assert os.path.isfile(ext_emb_path)

    # Load pretrained embeddings from file 找出包含在预训练中的所有词
    pretrained = set([
        line.rstrip().split()[0].strip()
        for line in codecs.open(ext_emb_path, 'r', 'utf-8')
        if len(ext_emb_path) > 0
    ])

    # We either add every word in the pretrained file,
    # or only words given in the `words` list to which
    # we can assign a pretrained embedding
    if chars is None:
        for char in pretrained:
            if char not in dictionary:
                dictionary[char] = 0
    else:
        for char in chars:
            #any() 函数用于判断给定的可迭代参数 iterable 是否全部为 False，则返回 False，如果有一个为 True，则返回 True。
            if any(x in pretrained for x in [
                char,
                char.lower(),
                re.sub('\d', '0', char.lower())
            ]) and char not in dictionary:
                dictionary[char] = 0

    word_to_id, id_to_word = create_mapping(dictionary)
    return dictionary, word_to_id, id_to_word


def save_maps(save_path, *params):
    """
    Save mappings and invert mappings
    """
    pass
    # with codecs.open(save_path, "w", encoding="utf8") as f:
    #     pickle.dump(params, f)


def load_maps(save_path):
    """
    Load mappings from the file
    """
    pass
    # with codecs.open(save_path, "r", encoding="utf8") as f:
    #     pickle.load(save_path, f)
