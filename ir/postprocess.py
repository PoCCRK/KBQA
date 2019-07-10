import numpy as np
import tensorflow as tf
import preprocess

class post_sq:
    def __init__(self, sq=preprocess.SimpleQuestions()):
        self.pre = sq
        self.base = sq.base
        self.knowledgebase = sq.knowledgebase
        self.words_dict = sq.words_dict
        self.knowledgebase_dict = sq.knowledgebase_dict
        self.local = True
        self.data = self.get_data("valid")
        #self.test = self.get_data("test")
        
    def valid(self):
        q_place = tf.placeholder(tf.int32,[1,20])
        a_place = tf.placeholder(tf.int32,[1,3])
        
        w_words = tf.Variable(tf.random_normal([len(self.words_dict), 128]), name="w_words")
        w_entities = tf.Variable(tf.random_normal([len(self.knowledgebase_dict), 128]), name="w_entities")
        
        e_t_q = tf.nn.embedding_lookup(w_words, q_place)
        e_t_c = tf.nn.embedding_lookup(w_entities, a_place)
        sum_q = tf.reduce_sum(e_t_q, axis=1)
        sum_c = tf.reduce_sum(e_t_c, axis=1)
        score = tf.matmul(sum_q, sum_c, transpose_b=True)
        
#         for j in range(2):
#             if j==0:
#                 questions, answers, candidates = self.data
#             else:
#                 questions, answers, candidates = self.test
            
        questions, answers, candidates = self.data
        TP = 0
        FP = 0
        FN = 0
        saver = tf.train.Saver()
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            saver.restore(sess, "./qa_base-1")
            #sess.run(init)
            variables_names = [v.name for v in tf.trainable_variables()]
            values = sess.run(variables_names)
            for k,v in zip(variables_names, values):
                print("Variable: ", k)
                print("Shape: ", v.shape)
                print(v)
                
            for i in range(len(questions)):
                question = questions[i]
                answer = answers[i]
                candidate = candidates[i]
                if len(candidate) <= 1:
                    continue
                scores = []
                for c_ in candidate:
                    r_s = sess.run(score, feed_dict={q_place:[question], a_place:[c_]})
                    scores.append(r_s[0][0])
                highest = max(scores)
                right_answer = []
                right_score = []
                result = []
                for i in range(len(scores)):
                    if scores[i] > highest - 0.6:
                        right_answer.append(candidate[i])
                        right_score.append(scores[i])
                
                if len(right_score) > 5:
                    for i in range(5):
                        highest = max(right_score)
                        result.append(right_answer[right_score.index(highest)])
                        del right_answer[right_score.index(highest)]
                        del right_score[right_score.index(highest)]
                else:
                    for i in range(len(right_score)):
                        highest = max(right_score)
                        result.append(right_answer[right_score.index(highest)])
                        del right_answer[right_score.index(highest)]
                        del right_score[right_score.index(highest)]
                
                print("num: " + str(len(candidate)) + "\t" +str(len(result)) )
                if answer in result:
                    TP = TP + 1
                else:
                    FN = FN + 1
                for a_ in result:
                    if a_[1] != answer[1]:
                        FP = FP + 1
                        break
                
        print(TP,FN,FP)
        return TP,FN,FP
    
    def get_data(self, dataset):
        questions = []
        answers = []
        candidates = []
        if self.local:
            with open("D:/QA/wikidata-simplequestions-master/local/local_valid_data", "r", encoding='UTF-8') as file:
                for line in file:
                    line = line.strip().split("\t")
                    question = []
                    answer = []
                    candidate = []
                    for q in line[0].split():
                        question.append(int(q))
                    for a in line[1].split():
                        answer.append(int(a))
                    line = line[2:]
                    for l in line:
                        c = []
                        for c_ in l.split():
                            c.append(int(c_))
                        candidate.append(np.array(c))
                     
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
            questions.append(np.array(question))
            
            answer = []
            for e in s,p,o:
                try:
                    answer.append(self.knowledgebase_dict.index(e))
                except ValueError:
                    answer.append(0)
            answers.append(np.array(answer))
            
            candidate = []
            for spo in self.knowledgebase:
                if line["subject"] in spo:
                    c = []
                    for e in spo:
                        try:
                            c.append(self.knowledgebase_dict.index(e))
                        except ValueError:
                            c.append(0)
                    candidate.append(c)
            candidates.append(np.array(candidate))
        
        questions = np.array(questions)
        answers = np.array(answers)
        candidates = np.array(candidates)
        with open("D:/QA/wikidata-simplequestions-master/local/local_valid_data", "w", encoding='UTF-8') as file:
            for ii in range(len(questions)):
                for w in questions[ii]:
                    file.write(str(w)+" ")
                file.write("\t")
                for a in answers[ii]:
                    file.write(str(a)+" ")
                file.write("\t")
                candidate = candidates[ii]
                for c in candidate:
                    for c_ in c:
                        file.write(str(c_)+" ")
                    file.write("\t")
                file.write("\n")
        return questions, answers, candidates
    

if __name__ == "__main__":
    print("postprocess.py")
    post_sq = post_sq()
    q,a,c = post_sq.data
    print(np.shape(q), np.shape(a), np.shape(c))
    post_sq.valid()

# (2013, 20) (2013, 3) (2013,)
# 1083 857 1815 125    2231 1874 3817 288
