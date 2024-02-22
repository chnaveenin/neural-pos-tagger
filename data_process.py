import conllu

class DataProcess():
    def __init__(self, file_path='', flag=False, prev=0, succ=0):
        self.data = self.read_file(file_path)
        self.flag = flag
        self.prev = prev
        self.succ = succ
        self.final_data, self.word_to_ix, self.tag_to_ix = self.get_words_and_tags(self.data)
        
    def read_file(self, file_path):
        with open(file_path, "r", encoding="utf-8") as file:
            data = file.read()
            data = conllu.parse(data)
            data = self.extract_columns(data)
        return data
        
    def extract_columns(self, data):
        result = []
        for sentence in data:
            sent_tokens = []
            for token in enumerate(sentence):
                sent_tokens.append((token[0], token[1]["form"], token[1]["upos"]))
            result.append(sent_tokens)
        return result

    def get_words_and_tags(self, data):
        all_words = []
        for sentence in data:
            for token in sentence:
                all_words.append(token[1])

        # add all tags into a list
        all_tags = []
        for sentence in data:
            for token in sentence:
                all_tags.append(token[2])
        # create tag_index dictionary
        tag_to_ix = {}
        for tag in all_tags:
            if tag not in tag_to_ix:
                tag_to_ix[tag] = len(tag_to_ix)

        word_to_ix = {}
        for word in all_words:
            if word not in word_to_ix:
                word_to_ix[word] = len(word_to_ix)
        word_to_ix["<UNK>"] = len(word_to_ix)
        if self.flag:
            word_to_ix["<START>"] = len(word_to_ix)
            word_to_ix["<END>"] = len(word_to_ix)

        final_data = []
        # print(data)
        for sentence in (data):
            # print(sentence)
            sentence_temp = []
            if self.flag:
                for i in range(self.prev):
                    sentence_temp.append((word_to_ix["<START>"], tag_to_ix["DET"]))
            for token in sentence:
                token = (word_to_ix[token[1]], tag_to_ix[token[2]])
                sentence_temp.append(token)
            if self.flag:
                for i in range(self.succ):
                    sentence_temp.append((word_to_ix["<END>"], tag_to_ix["DET"]))
            final_data.append(sentence_temp)
        # print(final_data)
        return final_data, word_to_ix, tag_to_ix
    
    def get_data_from_prev(self, path):
        data = []
        dev_data = self.read_file(path)
        for sentence in dev_data:
            temp_sentence = []
            if self.flag:
                for i in range(self.prev):
                    temp_sentence.append((self.word_to_ix["<START>"], self.tag_to_ix["DET"]))
            for word in sentence:
                if word[1] in self.word_to_ix.keys():
                    index = self.word_to_ix[word[1]]
                    if word[2] in self.tag_to_ix.keys():
                        index2 = self.tag_to_ix[word[2]]
                        temp_sentence.append((index, index2))
                    else:
                        index2 = self.tag_to_ix["DET"]
                        temp_sentence.append((index, index2))
                else:
                    index = self.word_to_ix["<UNK>"]
                    if word[2] in self.tag_to_ix.keys():
                        index2 = self.tag_to_ix[word[2]]
                        temp_sentence.append((index, index2))
                    else:
                        index2 = self.tag_to_ix["DET"]
                        temp_sentence.append((index, index2))
            if self.flag:
                for i in range(self.succ):
                    temp_sentence.append((self.word_to_ix["<END>"], self.tag_to_ix["DET"]))
            data.append(temp_sentence)
        return data