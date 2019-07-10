import numpy as np
import random
from qwikidata.linked_data_interface import get_entity_dict_from_api

class SimpleQuestions:
# Question Answering Benchmarks for Wikidata 
# by Diefenbach, Dennis and Tanon, Thomas and Singh, Kamal and Maret, Pierre
# ISWC 2017
    #如果操作不当可能，会有成员初始化失败的bug，自己使用就不管完备性问题了
    def __init__(self):
        self.local = True
        self.file_path = {"test": "D:/QA/wikidata-simplequestions-master/annotated_wd_data_test_answerable.txt",
                          "train": "D:/QA/wikidata-simplequestions-master/annotated_wd_data_train_answerable.txt",
                          "valid": "D:/QA/wikidata-simplequestions-master/annotated_wd_data_valid_answerable.txt"}
        self.base = self.get_base()
        self.knowledgebase = self.get_knowledgebase()
        self.knowledgebase_dict = self.get_knowledgebase_dict()
        self.words_dict = self.get_words_dict()
        self.data = self.get_data("train")
        self.pretrain_data = self.get_pretrain_data()
    
    def get_words_dict(self):
        buf = ""
        for dataset in self.base:
            for line in self.base[dataset]:
                buf += line["qText"] + " "
        words = buf.split()
        words = list(set(words))
        words.sort()
        words.insert(0, "UNKNOWN")
        return words
    
    def get_knowledgebase_dict(self):
        items = []
        for line in self.knowledgebase:
            items.extend(line)
        items = list(set(items))
        items.sort()
        items.insert(0, "UNKNOWN")
        return items
    
    def get_knowledgebase(self):
        KB = []
        if self.local:
            with open("D:/QA/wikidata-simplequestions-master/local_knowledgebase", "r", encoding='UTF-8') as file:
                for line in file:
                    KB.append(line.strip().split("\t"))
            return KB
        
        for dataset in self.base:
            for line in self.base[dataset]:
                s = line["subject"]
                p = line["predicate"]
                o = line["object"]
                if p[0]=="R":
                    p = p.replace("R","P")
                    s = line["object"]
                    o = line["subject"]
                if [s, p, o] not in KB:
                    KB.append([s, p, o])
                    print(s,p,o)
        
        with open("D:/QA/wikidata-simplequestions-master/local/local_knowledgebase", "w", encoding='UTF-8') as file:
            for line in KB:
                file.write(line[0] + "\t" + line[1] + "\t" + line[2] + "\n")
        return KB
    
    def get_data(self, dataset):
        questions = []
        answers = []
        candidates = []
        if dataset=="train":
            file_path = "D:/QA/wikidata-simplequestions-master/local/local_train_data"
        elif dataset=="valid":
            file_path = "D:/QA/wikidata-simplequestions-master/local/local_valid_data"
        elif dataset=="test":
            file_path = "D:/QA/wikidata-simplequestions-master/local/local_test_data"
        if self.local:
            with open(file_path, "r", encoding='UTF-8') as file:
                for line in file:
                    line = line.strip().split("\t")
                    question = []
                    answer = []
                    candidate = []
                    for q in line[0].split():
                        question.append(int(q))
                    for a in line[1].split():
                        answer.append(int(a))
                    for c in line[2].split():
                        candidate.append(int(c))
                           
                    questions.append(np.array(question))
                    answers.append(np.array(answer))
                    candidates.append(np.array(candidate))
            return np.array(questions), np.array(answers), np.array(candidates)
        
        for line in self.base[dataset]:
            s = line["subject"]
            p = line["predicate"]
            o = line["object"]
            if p[0] == "R":
                p = p.replace("R","P")
                s = line["object"]
                o = line["subject"]
            question = []
            words = line["qText"].split()
            # 忽略长度大于20的问题
            if len(words) > 20:
                continue
            for word in words:
                try:
                    question.append(self.words_dict.index(word))
                except ValueError:
                    question.append(0)
            while len(question) < 20:
                question.append(-1)
            
            answer = []
            for e in s,p,o:
                try:
                    answer.append(self.knowledgebase_dict.index(e))
                except ValueError:
                    answer.append(0)
            
            for spo in self.knowledgebase:
                if line["subject"] in spo and [s,p,o]!=spo:
                    candidate = []
                    for e in spo:
                        try:
                            candidate.append(self.knowledgebase_dict.index(e))
                        except ValueError:
                            candidate.append(0)
                    questions.append(np.array(question))
                    answers.append(np.array(answer))
                    candidates.append(np.array(candidate))
        
        questions = np.array(questions)
        answers = np.array(answers)
        candidates = np.array(candidates)
        
#         index = [i for i in range(len(questions))] 
#         random.shuffle(index)
#         questions = questions[index]
#         answers = answers[index]
#         candidates = candidates[index]
        
        with open(file_path, "w", encoding='UTF-8') as file:
            for ii in range(len(questions)):
                for w in questions[ii]:
                    file.write(str(w)+" ")
                file.write("\t")
                for a in answers[ii]:
                    file.write(str(a)+" ")
                file.write("\t")
                for c in candidates[ii]:
                    file.write(str(c)+" ")
                #file.write("\t")
                file.write("\n")
        return questions, answers, candidates
    
    def get_pretrain_data(self):
        s = []
        p = []
        o = []
        if self.local:
            with open("D:/QA/wikidata-simplequestions-master/local/local_sq_pre_data", "r", encoding='UTF-8') as file:
                for line in file:
                    line = line.split("\t")
                    s.append(int(line[0]))
                    p.append(int(line[1]))
                    o.append(int(line[2]))
                return np.array(s),np.array(p),np.array(o)
        for spo in self.knowledgebase:
            try:
                s_ = self.knowledgebase_dict.index(spo[0])
                p_ = self.knowledgebase_dict.index(spo[1])
                o_ = self.knowledgebase_dict.index(spo[2])
            except ValueError:
                continue
            s.append(s_)
            p.append(p_)
            o.append(o_)
        s = np.array(s)
        p = np.array(p)
        o = np.array(o)
        #打乱数据
        index = [i for i in range(len(s))] 
        random.shuffle(index)
        s = s[index]
        p = p[index]
        o = o[index]
        with open("D:/QA/wikidata-simplequestions-master//local/local_sq_pre_data", "w", encoding='UTF-8') as file:
            for ii in range(len(s)):
                file.write(str(s[ii])+"\t"+str(p[ii])+"\t"+str(o[ii])+"\n")
        return s, p, o
    
    def get_base(self):
        base = {"test":[], "train":[], "valid":[]}
        for dataset in base:
            with open(self.file_path[dataset], "r", encoding='UTF-8') as file:
                for line in file:
                    line = line.strip()
                    buf = line.split("\t")
                    
                    base[dataset].append({})
                    base[dataset][-1]["subject"] = buf[0]
                    base[dataset][-1]["predicate"] = buf[1]
                    base[dataset][-1]["object"] = buf[2]
                    
                    question = buf[3]
                    question = question.lower()
                    question = question.rstrip("?")
                    
                    if question[0:5]=="name ":
                        question = question.rstrip(".")
                    if question[0:7]=="what's ":
                        question = question.replace("what's", "what is", 1)
                    if question[0:6]=="who's ":
                        question = question.replace("who's", "who is", 1)
                    if "(" in question and ")" in question:
                        question = question.replace("(", " ").replace(")", " ")
                    if "http:" not in question and "https:" not in question:
                        question = question.replace("/", " / ").replace("?", " ? ")
                    if "\'s " in question:
                        question = question.replace("\'s ", " \'s ")
                    question = question.replace("\"", " ")
                    question = question.replace(",", " , ").replace(":", " : ").replace("!", " ! ")
                    # 部分特殊符号已经在文件中手动修改
                    base[dataset][-1]["qText"] = question
        
        return base



if __name__ == "__main__":
    print("preprocess.py")
    sq = SimpleQuestions()
    print(len(sq.knowledgebase))
    print(len(sq.words_dict))
    print(len(sq.knowledgebase_dict))
    q,a,c = sq.data
    print(np.shape(q), np.shape(a), np.shape(c))
    p_s, p_p, p_o = sq.pretrain_data
    print(np.shape(p_s), np.shape(p_p), np.shape(p_o))

# 59436
# 16262
# 40431
# (115089, 20) (115089, 3) (115089, 3)
# (59436,) (59436,) (59436,)

# 59436
# 16262
# 40431
# (637806, 20) (637806, 3) (637806, 3)
# (59436,) (59436,) (59436,)
