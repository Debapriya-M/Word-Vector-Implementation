import tensorflow as tf

def cross_entropy_loss(inputs, true_w):
    """
    ==========================================================================

    inputs: The embeddings for context words. Dimension is [batch_size, embedding_size].
    true_w: The embeddings for predicting words. Dimension of true_w is [batch_size, embedding_size].

    Write the code that calculate A = log(exp({u_o}^T v_c))

    A = 


    And write the code that calculate B = log(\sum{exp({u_w}^T v_c)})


    B =

    ==========================================================================
    """

    transpose_matrix = tf.transpose(true_w)
    multi_matrix = tf.matmul(inputs, transpose_matrix)
    A = tf.log(tf.exp(tf.diag_part(multi_matrix)))
    #print(A)
    B = tf.log(tf.reduce_sum(tf.exp(multi_matrix), axis = 1))
    #print(B)


    return tf.subtract(B, A)

def nce_loss(inputs, weights, biases, labels, sample, unigram_prob):
    """
    ==========================================================================

    inputs: Embeddings for context words. Dimension is [batch_size, embedding_size].
    weigths: Weights for nce loss. Dimension is [Vocabulary, embeeding_size].
    biases: Biases for nce loss. Dimension is [Vocabulary, 1].
    labels: Word_ids for predicting words. Dimesion is [batch_size, 1].
    samples: Word_ids for negative samples. Dimension is [num_sampled].
    unigram_prob: Unigram probability. Dimesion is [Vocabulary].

    Implement Noise Contrastive Estimation Loss Here

    ==========================================================================
    """

    #tf_func.nce_loss(embed, nce_weights, nce_biases, train_labels, sample, unigram_prob)
    print("Entering the loss function ---->")
    print("unigram_prob : ", len(unigram_prob))
    print(weights)
    weight_rows, weight_columns = weights.get_shape()
    
    #u_o - to get the weight vector of the target word
    a = tf.nn.embedding_lookup(weights, labels)
    u_o = tf.reshape(a, [-1, weight_columns])
    print('Dimension of u_o : ', u_o)

    #b_o
    b_o = tf.nn.embedding_lookup(biases, labels)
    print('Dimension of b_o : ', b_o)

    #s(wo, wc) = (uTc uo) + bo
    multi_mat_o = tf.matmul(u_o, tf.transpose(inputs))
    print("Multiply: ", multi_mat_o)
    s_wo_wc = tf.add(tf.reshape(tf.diag_part(multi_mat_o), [-1,1]), b_o)
    print ("s_wo_wc: ",s_wo_wc)

    noise = 0.000003

    #k
    k = len(sample)
    print('The number of negative samples:', k)

    #P(w_o)
    prob_wo = tf.gather(unigram_prob, labels)
    pr_w_o = tf.log(tf.add(tf.math.scalar_mul(k, prob_wo), noise))
    print(pr_w_o)

    #sigma_o
    p_wo_wc = tf.sigmoid(tf.math.subtract(s_wo_wc, pr_w_o))
    print('p_wo_wc: ', p_wo_wc)

    #u_c
    # u_c = inputs
    # print('Dimension of u_c : ', u_c)

    #u_x
    u_x = tf.nn.embedding_lookup(weights, sample)
    print('Dimension of u_x : ', u_x)
   
    #b_x
    b_neg = tf.nn.embedding_lookup(biases, sample)
    tiled_bneg = tf.tile(b_neg, [2])
    b_x = tf.reshape(tiled_bneg, [-1, 1])
    print('Dimension of b_x : ', b_x)

    #s(wx, wc)
    multi_mat_x = tf.matmul(inputs, tf.transpose(u_x))
    s_wx_wc = tf.add(multi_mat_x, b_x)
    print ("s_wx_wc: ",s_wx_wc)

    #P(w_x)
    prob_wx = tf.gather(unigram_prob, sample)
    print('Shape: ', prob_wx)
    pr_w_x = tf.log(tf.add(tf.math.scalar_mul(k, tf.reshape(tf.tile(prob_wx, [2]), [-1, 1])), noise))
    print(pr_w_x)

    #sigma_x
    p_wx_wc = tf.sigmoid(tf.math.subtract(s_wx_wc, pr_w_x))
    print('p_wx_wc: ', p_wx_wc)


    #Calculation of first term
    sum3 = tf.add(p_wo_wc, noise)
    A = tf.log(sum3)
    print('A : ', A)


    #Calculation of second term
    dummy_scalar = tf.constant(1.000001, shape=[p_wx_wc.shape[0], p_wx_wc.shape[1]], dtype  = tf.float32)
    print('dummy_scalar: ' , dummy_scalar)
    value = tf.math.subtract(dummy_scalar, p_wx_wc)
    print('Value : ', value)
    B = tf.reshape(tf.reduce_sum(tf.log(value), axis = 1), [-1,1])
    print("B: ", B)

    nce_loss = tf.math.scalar_mul(-1.0, tf.math.add(A, B))
    print(nce_loss.shape)


    return nce_loss




