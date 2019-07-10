import tensorflow as tf
import numpy as np
from preprocess import SimpleQuestions
from postprocess import post_sq
import os

def embedding_q(q, w_q):
    # q: batch_size, 20
    # w_q: vocab_size, 128
    outputs = tf.nn.embedding_lookup(w_q, q)
    # outputs: batch_size, 20, 128
    return outputs

def embedding_a(a, w_a):
    # a: batch_size, 3
    # w_a: entities_size, 128
    outputs = tf.nn.embedding_lookup(w_a, a)
    # outputs: batch_size, 3, 128
    return outputs

def BiLSTM(e_qs, lstm_fw_cell, lstm_bw_cell):
    # e_qs: batch_size, timesteps=20, 128
    # Required shape: 'timesteps' tensors list of shape (batch_size, num_input)
    
    # Unstack to get a list of 'timesteps' tensors of shape (batch_size, num_input)
    qs = tf.unstack(e_qs, 20, axis=1)
    
    outputs, _, _ = tf.contrib.rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, qs, dtype=tf.float32)
    # outputs: a list of 'timesteps'=20 tensors of shape (batch_size, num_input)
    return outputs

def ATT(lstm_qs, e_as, w, b):
    # lstm_qs: a list of 'timesteps'=20 tensors of shape (batch_size, num_input)
    # e_as: batch_size, 3, d=128
    ea = tf.unstack(e_as, 3, axis=1) # 3, batch_size, 128
    
    # 计算权重
    e_s = ea[0] # 1, 128
    e_p = ea[1]
    e_o = ea[2]
    q_ss = []
    q_ps = []
    q_os = []
    for i in range(20):
        w_is = tf.matmul(tf.tanh(tf.concat([lstm_qs[i],e_s], axis=1)), w_att, transpose_b=False) + b_att
        w_ip = tf.matmul(tf.tanh(tf.concat([lstm_qs[i],e_p], axis=1)), w_att, transpose_b=False) + b_att
        w_io = tf.matmul(tf.tanh(tf.concat([lstm_qs[i],e_o], axis=1)), w_att, transpose_b=False) + b_att
        # 关键的权重a的计算，
        a_is = tf.exp(w_is) / (tf.exp(w_is) + tf.exp(w_ip) + tf.exp(w_io))
        a_ip = tf.exp(w_ip) / (tf.exp(w_is) + tf.exp(w_ip) + tf.exp(w_io))
        a_io = tf.exp(w_io) / (tf.exp(w_is) + tf.exp(w_ip) + tf.exp(w_io))
        q_is = tf.nn.relu(a_is * lstm_qs[i])
        q_ip = tf.nn.relu(a_ip * lstm_qs[i])
        q_io = tf.nn.relu(a_io * lstm_qs[i])
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
    
    return tf.stack((q_s, q_p, q_o), axis=1) # batch_size, 3, 128

def loss(att_as, att_cs, e_as, e_cs):
    # att_as: batch_size, 3, d=128
    # e_as: batch_size, 3, d=128
    bs = tf.shape(e_as)[0]
    
    def cond(i, att_a_s, att_c_s, a_s, c_s, loss):
        return i<bs
    def body(i, att_a_s, att_c_s, a_s, c_s, loss):
        p_prediction = tf.matmul([att_a_s[i][0]], [a_s[i][0]], transpose_b=True) + tf.matmul([att_a_s[i][1]], [a_s[i][1]], transpose_b=True) + tf.matmul([att_a_s[i][2]], [a_s[i][2]], transpose_b=True)
        n_prediction = tf.matmul([att_c_s[i][0]], [c_s[i][0]], transpose_b=True) + tf.matmul([att_c_s[i][1]], [c_s[i][1]], transpose_b=True) + tf.matmul([att_c_s[i][2]], [c_s[i][2]], transpose_b=True)
        loss = loss + tf.maximum(0., 0.6 - p_prediction[0][0] + n_prediction[0][0])
        i = i + 1
        return i, att_a_s, att_c_s, a_s, c_s, loss
    
    _, _, _, _, _, r_loss = tf.while_loop(cond, body, [0, att_as, att_cs, e_as, e_cs, 0.])
    return r_loss

def pre_loss(s_s, p_s, o_s):
    # batch_size, 128
    bs = tf.shape(s_s)[0]
    def cond_1(i, s, p, p_p, o_s, loss):
        return i<bs
    def body_1(i, s, p, p_p, o_s, loss):
        # p_p = positive_prediction
        n_prediction = tf.reduce_sum(tf.square(s + p - o_s[i]))
        loss = loss + tf.maximum(0., 1 + p_p - n_prediction)
        i = i + 1
        return i, s, p, p_p, o_s, loss
    def cond_2(i, s_s, p_s, o_s, loss):
        return i<bs
    def body_2(i, s_s, p_s, o_s, loss):
        p_prediction = tf.reduce_sum(tf.square(s_s[i] + p_s[i] - o_s[i]))
        _, _, _, _, _, r_loss = tf.while_loop(cond_1, body_1, [0, s_s[i], p_s[i], p_prediction, o_s, 0.])
        loss = loss + r_loss
        i = i + 1
        return i, s_s, p_s, o_s, loss
    
    _, _, _, _, r_loss = tf.while_loop(cond_2, body_2, [0, s_s, p_s, o_s, 0.])
    return r_loss

if __name__ == "__main__":
    print("base.py")
    is_trained = False
    batch_size = 128
    epoch_num = 5
    
    sq = SimpleQuestions()
    post_sq = post_sq(sq)
    t_q,t_a,t_c = post_sq.data
    q,a,c = sq.data
    s,p,o = sq.pretrain_data
    
    batch_num = len(q) // batch_size + 1
    pre_batch_num = len(s) // batch_size + 1
    
    # define graph in tensorflow
    w_words = tf.Variable(tf.random_normal([len(sq.words_dict), 128]), name="w_words")
    w_entities = tf.Variable(tf.random_normal([len(sq.knowledgebase_dict), 128]), name="w_entities")
    w_att = tf.Variable(tf.random_normal([128*2, 1]), name="w_att")
    b_att = tf.Variable(0.1, dtype=tf.float32, name="b_att")
    # hidden unit of lstm is 64 (half of the embedding=128)
    lstm_fw = tf.nn.rnn_cell.LSTMCell(64, name="lstm_fw", reuse=tf.AUTO_REUSE)
    lstm_bw = tf.nn.rnn_cell.LSTMCell(64, name="lstm_bw", reuse=tf.AUTO_REUSE)
    
    dataset = tf.data.Dataset.from_tensor_slices((q,a,c))
    dataset = dataset.batch(batch_size).repeat()
    pre_dataset = tf.data.Dataset.from_tensor_slices((s,p,o))
    pre_dataset = pre_dataset.shuffle(buffer_size=100000, reshuffle_each_iteration=True).batch(batch_size).repeat()
    iterator = dataset.make_one_shot_iterator()
    next_q, next_a, next_c = iterator.get_next()
    pre_iterator = pre_dataset.make_one_shot_iterator()
    pre_next_s, pre_next_p, pre_next_o = pre_iterator.get_next()
    
    e_ss = embedding_a(pre_next_s, w_entities)
    e_ps = embedding_a(pre_next_p, w_entities)
    e_os = embedding_a(pre_next_o, w_entities)
    r_pre_loss = pre_loss(e_ss, e_ps, e_os)
    pre_train_step = tf.train.AdamOptimizer().minimize(r_pre_loss)
    
    r_e_qs = embedding_q(next_q, w_words)
    r_e_as = embedding_a(next_a, w_entities)
    r_e_cs = embedding_q(next_c, w_entities)
    r_lstm = BiLSTM(r_e_qs, lstm_fw, lstm_bw)
    r_att_as = ATT(r_lstm, r_e_as, w_att, b_att)
    r_att_cs = ATT(r_lstm, r_e_cs, w_att, b_att)
    r_loss = loss(r_att_as, r_att_cs, r_e_as, r_e_cs)
    r_train_step = tf.train.AdamOptimizer().minimize(r_loss)
    
    tf.add_to_collection("losses", tf.contrib.layers.l2_regularizer(0.1)(w_entities))
    tf.add_to_collection("losses", tf.contrib.layers.l2_regularizer(0.1)(w_words))
    tf.add_to_collection("losses", tf.contrib.layers.l2_regularizer(0.1)(w_att))
    tf.add_to_collection("losses", r_loss)
    loss = tf.add_n(tf.get_collection("losses"))
    train_step = tf.train.AdamOptimizer().minimize(loss)
    
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
        sess.run(init)
        with open("D:/homework/QA/ir/loss.txt", "w", encoding='UTF-8') as file:
            for e in range(epoch_num):
                print("---------------- epoch_num: " + str(e) + " -----------------------")
                file.write("epoch_num: " + str(e) + "\n")
                print("---------------------train--------------------")
                for b in range(batch_num):
#                     _, status, r_status= sess.run([train_step, loss, r_loss])
#                     print(status, r_status)
#                     file.write(str(status) + " " + str(r_status) + "\t")
                    _, r_status= sess.run([r_train_step,  r_loss])
                    print(r_status)
                    file.write(str(r_status) + "\t")
#                 if e==0:
#                     print("-------------------pre-train------------------")
#                     for b in range(pre_batch_num):
#                         _, status = sess.run([pre_train_step, r_pre_loss])
#                         print(status)
                
                TP = 0
                FN = 0
                FP = 0
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
                    
                    result = np.array(result)
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
                file.write("\n")
                file.write(str((TP,FN,FP))+"\n")
                file.write("\n")
                saver.save(sess,'./qa_ir', global_step=e)
                
        print("Training done. Model saved at qa_ir")
        
        os.system("shutdown -s -t 60")

