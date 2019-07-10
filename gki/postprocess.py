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
        self.data = self.get_data("valid") # 776 43 799 20
        
    def valid(self):
        
        t_q, t_a, t_c = self.data
        
        w_words = tf.Variable(tf.random_normal([len(self.words_dict), 128]), name="w_words")
        w_entities = tf.Variable(tf.random_normal([len(self.knowledgebase_dict), 128]), name="w_entities")
        w_att = tf.Variable(tf.random_normal([128*2, 1]), name="w_att")
        b_att = tf.Variable(0.1, dtype=tf.float32, name="b_att")
        lstm_fw = tf.nn.rnn_cell.LSTMCell(64, name="lstm_fw", reuse=tf.AUTO_REUSE)
        lstm_bw = tf.nn.rnn_cell.LSTMCell(64, name="lstm_bw", reuse=tf.AUTO_REUSE)
        
        q_place = tf.placeholder(tf.int32,[1,20])
        a_place = tf.placeholder(tf.int32,[1,3])
        embed_q = tf.nn.embedding_lookup(w_words, q_place) # 1, 20, 128
        embed_c = tf.nn.embedding_lookup(w_entities, a_place) # 1, 3, 128
        un_q = tf.unstack(embed_q, 20, axis=1) # 20, 1, 128
        un_c = tf.unstack(embed_c, 3, axis=1) # 3, 1, 128
        t_lstm, _, _ = tf.contrib.rnn.static_bidirectional_rnn(lstm_fw, lstm_bw, un_q, dtype=tf.float32) # 20, 1, 128
        e_s = un_c[0] # 1, 128
        e_p = un_c[1]
        e_o = un_c[2]
        q_ss = []
        q_ps = []
        q_os = []
        for i in range(20):
            w_is = tf.matmul(tf.tanh(tf.concat([t_lstm[i],e_s], axis=1)), w_att, transpose_b=False) + b_att
            w_ip = tf.matmul(tf.tanh(tf.concat([t_lstm[i],e_p], axis=1)), w_att, transpose_b=False) + b_att
            w_io = tf.matmul(tf.tanh(tf.concat([t_lstm[i],e_o], axis=1)), w_att, transpose_b=False) + b_att
            # 关键的权重a的计算，
            a_is = tf.exp(w_is) / (tf.exp(w_is) + tf.exp(w_ip) + tf.exp(w_io))
            a_ip = tf.exp(w_ip) / (tf.exp(w_is) + tf.exp(w_ip) + tf.exp(w_io))
            a_io = tf.exp(w_io) / (tf.exp(w_is) + tf.exp(w_ip) + tf.exp(w_io))
            q_is = tf.nn.relu(a_is * t_lstm[i])
            q_ip = tf.nn.relu(a_ip * t_lstm[i])
            q_io = tf.nn.relu(a_io * t_lstm[i])
            q_ss.append(q_is)
            q_ps.append(q_ip)
            q_os.append(q_io)
        
        q_s = q_ss[0]
        q_p = q_ps[0]
        q_o = q_os[0]
        for i in range(1, 20):
            q_s = q_s + q_ss[i]
            q_p = q_p + q_ps[i]
            q_o = q_o + q_os[i]
        
        score = tf.matmul(q_s, e_s, transpose_b=True) + tf.matmul(q_p, e_p, transpose_b=True) + tf.matmul(q_o, e_o, transpose_b=True)
        
        saver = tf.train.Saver()
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            saver.restore(sess, "./qa_ir-4")
            #sess.run(init)
            variables_names = [v.name for v in tf.trainable_variables()]
            values = sess.run(variables_names)
            for k,v in zip(variables_names, values):
                print("Variable: ", k)
                print("Shape: ", v.shape)
                print(v)
            right = 0
            wrong = 0
            right_ = 0
            wrong_ = 0
            for i in range(len(t_q)):
                question = t_q[i]
                answer = t_a[i]
                candidate = t_c[i]
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
                
                #print("num: " + str(len(candidate)) + "\t" +str(len(result)) )
                right_flag = 0
                for a_ in result:
                    if a_[0]==answer[0] and a_[1]==answer[1] and a_[2]==answer[2]:
                        right_flag = 2
                        break
                    if a_[1] == answer[1]:
                        right_flag = 1
                if right_flag == 0:
                    wrong = wrong + 1
                    wrong_ = wrong_ + 1
                elif right_flag == 1:
                    right_ = right_ + 1
                    wrong = wrong + 1
                elif right_flag == 2:
                    right = right + 1
                    right_ = right_ + 1
                
            print(right,wrong,right_,wrong_)
            
        return right,wrong,right_,wrong_
    
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

# 1158 782 1738 202
# 1165 775 1772 168
# 1155 785 1750 190
# 1174 766 1783 157
# 1184 756 1796 144