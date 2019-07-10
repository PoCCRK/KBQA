import numpy as np
import tensorflow as tf
from preprocess import SimpleQuestions
from postprocess import post_sq

def embedding_q(q, w_q):
    # q: batch_size, 20
    # w_q: vocab_size, d=128
    outputs = tf.nn.embedding_lookup(w_q, q)
    # outputs: batch_size, 20, d=128
    mean, variance = tf.nn.moments(w_q, [0,0])
    return (outputs - mean) / tf.sqrt(variance)

def embedding_a(a, w_a):
    # a: batch_size, 3
    # w_a: entities_size, d=128
    outputs = tf.nn.embedding_lookup(w_a, a)
    # outputs: batch_size, 3, d=128
    mean, variance = tf.nn.moments(w_a, [0,0])
    return (outputs - mean) / tf.sqrt(variance)

def loss(e_qs, e_as, e_cs):
    # e_qs: batch_size, 20, d=128
    # e_as: batch_size, 3, d=128
    # e_cs: batch_size, 3, d=128
    bs = tf.shape(e_qs)[0]
    
    def cond(i, e_qs, e_as, e_cs, loss):
        return i<bs
    def body(i, e_qs, e_as, e_cs, loss):
        sq = tf.reduce_sum(e_qs[i], axis=0, keep_dims=True)
        sa = tf.reduce_sum(e_as[i], axis=0, keep_dims=True)
        sc = tf.reduce_sum(e_cs[i], axis=0, keep_dims=True)
        p_prediction = tf.matmul(sq, sa, transpose_b=True)
        n_prediction = tf.matmul(sq, sc, transpose_b=True)
        loss = loss + tf.maximum(0., 0.6 - p_prediction[0][0] + n_prediction[0][0])
        i = i + 1
        return i, e_qs, e_as, e_cs, loss
    
    _, _, _, _, r_loss = tf.while_loop(cond, body, [0, e_qs, e_as, e_cs, 0.])
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
    epoch_num = 5
    batch_size = 128
    
    sq = SimpleQuestions()
    post_sq = post_sq(sq)
    t_q, t_a, t_c = post_sq.data
    q,a,c = sq.data
    s,p,o = sq.pretrain_data
    
    batch_num = len(q) // batch_size + 1
    pre_batch_num = len(s) // batch_size + 1
    
    q_place = tf.placeholder(tf.int32,[1,20])
    a_place = tf.placeholder(tf.int32,[1,3])
    
    # define graph in tensorflow
    w_words = tf.Variable(tf.random_uniform([len(sq.words_dict), 128], dtype=tf.float32), name="w_words")
    w_entities = tf.Variable(tf.random_uniform([len(sq.knowledgebase_dict), 128], dtype=tf.float32), name="w_entities")
    
    dataset = tf.data.Dataset.from_tensor_slices((q,a,c))
    dataset = dataset.batch(batch_size).repeat()
    pre_dataset = tf.data.Dataset.from_tensor_slices((s,p,o))
    pre_dataset = pre_dataset.batch(batch_size).repeat()
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
    r_e_cs = embedding_a(next_c, w_entities)
    r_loss = loss(r_e_qs, r_e_as, r_e_cs)
    r_train_step = tf.train.AdamOptimizer().minimize(r_loss)
    
    e_t_q = tf.nn.embedding_lookup(w_words, q_place)
    e_t_c = tf.nn.embedding_lookup(w_entities, a_place)
    sum_q = tf.reduce_sum(e_t_q, axis=1)
    sum_c = tf.reduce_sum(e_t_c, axis=1)
    score = tf.matmul(sum_q, sum_c, transpose_b=True)
    
    tf.add_to_collection("losses", tf.contrib.layers.l2_regularizer(0.1)(w_entities))
    tf.add_to_collection("losses", tf.contrib.layers.l2_regularizer(0.1)(w_words))
    tf.add_to_collection("losses", r_loss)
    loss = tf.add_n(tf.get_collection("losses"))
    train_step = tf.train.AdamOptimizer().minimize(loss)
    
    saver = tf.train.Saver(max_to_keep=5)
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        #saver.restore(sess, "./qa_base")
        with open("D:/homework/QA/base/loss.txt", "w", encoding='UTF-8') as file:
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
                
#                 TP = 0
#                 FP = 0
#                 FN = 0
#                 for i in range(len(t_q)):
#                     question = t_q[i]
#                     answer = t_a[i]
#                     candidate = t_c[i]
#                     if len(candidate) <= 1:
#                         continue
#                     scores = []
#                     for c_ in candidate:
#                         r_s = sess.run(score, feed_dict={q_place:[question], a_place:[c_]})
#                         scores.append(r_s[0][0])
#                     
#                     highest = max(scores)
#                     right_answer = []
#                     right_score = []
#                     result = []
#                     
#                     for i in range(len(scores)):
#                         if scores[i] > highest - 0.6:
#                             right_answer.append(candidate[i])
#                             right_score.append(scores[i])
#                     
#                     if len(right_score) > 5:
#                         for i in range(5):
#                             highest = max(right_score)
#                             result.append(right_answer[right_score.index(highest)])
#                             del right_answer[right_score.index(highest)]
#                             del right_score[right_score.index(highest)]
#                     else:
#                         for i in range(len(right_score)):
#                             highest = max(right_score)
#                             result.append(right_answer[right_score.index(highest)])
#                             del right_answer[right_score.index(highest)]
#                             del right_score[right_score.index(highest)]
#                     
#                     print("num: " + str(len(candidate)) + "\t" +str(len(result)) )
#                     if answer is in result:
#                         TP = TP + 1
#                     else:
#                         FN = FN + 1
#                     for a_ in result:
#                         if a_[1] != answer[1]:
#                             FP = FP + 1
#                             break
#                     
#                 print(TP,FN,FP)
#                 file.write("\n")
#                 file.write(str((TP,FN,FP))+"\n")
                file.write("\n")
                saver.save(sess,'./qa_base', global_step=e)
                
        print("Training done. Model saved at qa_base")
    

