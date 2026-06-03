"""
Problems with RNN:
- Early words may lose influence over long sequences. GRU mitigates this partially.
- Sequential processing, can't be parallelized.


Both stem from same fundamental issue. Each hidden reperesentation is a function of previous one.

- What if we compute how relevant each word is from every other word in a sequence?
- Relevance or similarity in attention is computed using dot product.
    - Cosine similarity is not used, at it normalizes.
    - Only preserves direction, not magnitude. So, how similar? is not preserved.
    - dot product is much faster as well.

Q: But, is this enough? Learning just the similarity?
A: No, we also need to learn how much to attend each word.

Other concerns:
- Words change their meaning based on their context.
- If AI processes words in isolation, concept can be confused.
- Attention is the mechanism, that allows words to look around at the other words in the sentence and absorb their context.


# Query, Key and Value (Q,K,V):
    Query: The word we are trying to find the relavance of.
    Key: The word we are trying to find the relevance to.
    Value: The actual information we want to extract from the key.
        - Key is usually a concise/summarized representation of the value.
        - Similarity between query and key is determined by dot product.
        - Only similar keys with similar values will be attended to.

# Intuition:
Take a sentence: "The cat didn't cross the street because it was too tired."
- The word "it" is ambiguous. Isolated, it could refer to cat, street.
- For "it" to represent it's true meaning, it needs to be aware of other words in the context.

Step 1: For being able to represent it's own true meaning:
    - Query: "it" asks a question => What pronouns, representing a singular noun is capable of being tired?
    - Key:
        "street": has some value, but key is an easily searchable repr. Say, it says "I'm a singular noun, but not cabaple of being tired"
        "cat": Say, it says "I'm a singular noun, and cabaple of being tired"

Step 2: Calculating Attention Score:
- "it" match with "cross" : 0%
- "it" match with "street": 10%
- "it" match with "cat": 90%

Step 3: What this means is, "it" should have a value, that's weighted average of values of (not key) "street" and "cat",
    but more weighted towards "cat". So, the value of "it" is mostly influenced by "cat", and a little or negligilbe by "street".


- This happens parallely for all words.
- KV-caching is used to speed up the process.

-------------------------------
MATH
-------------------------------

Let's say we have 3 weight matrices. W_Q, W_K, W_V. These
These are learnable parameters that are used to transform the input word embeddings into query, key and value vectors.
q = W_Q * e
k = W_K * e
v = W_V * e

(here, this process is essentially finding out a good question for the current word. And key,value too)
When processing whole sentence at once, q,k,v become Q,K,V.

Step1: Calculating Attention Score:
    Raw attention score= Q * K^T
    (Q is of shape [seq_len, d_k], K^T is of shape [d_k, seq_len], so the result is of shape [seq_len, seq_len])
        d_k is the dimension of the key vectors.

Step2: Scaling the attention score:
    - Without scaling, dot product can be huge, resulting in gradients vanishing to zero. in the downstream softmax function.
    - So, we scale the attention score by dividing it by sqrt(d_k).
    i.e : scaled attention score = (Q * K^T) / sqrt(d_k)

    - This keeps the variance of numbers stable. Usually around 1.

Step3: Applying Softmax, scores to percentages:
    - At this point, score is a matrix of shape [seq_len, seq_len], full of random numbers.
    - We need these to be interpretable as probabilities.
    - We apply softmax function at every row of the matrix.
    - Attention weights = softmax(scaled attention score)
    i.e: attention_weight[i] = softmax(scaled_attention_score[i]) = softmax((Q * K^T)[i] / sqrt(d_k))

    say, to start with we had [4.2, 0.1, 0.5, 2.5, -1.5]
    After scaling, we get [2.1, 0.05, 0.25, 1.25, -0.75]
    After applying softmax, we get [0.45, 0.1, 0.15, 0.2, 0.1] => sums to 1.

Step4: Weighted sum of values:
    - Now, we need to use these attention weights to get a weighted sum of the value vectors.
    - Output = attention_weights * V


This makes the final equation as:
    Attention(Q, K, V) = softmax((Q * K^T) / sqrt(d_k)) * V


----
Example sentence: "The trophy didn't fit in the suitcase because it was too big."

Questions:
Q: Why do we need Q,K,V? Why not just use embeddings directly?
A:
    Problem 1: say every word is represented as embedding. (e_i for query and e_j for key).
        - similarity would be e_i . e_j
        - Dot product is symmetric, so, e_i . e_j = e_j . e_i
        - But, this is wrong. "it" in above context is similar to "trophy", but "trophy" is not similar to "it". "trophy" doesn't need to attend  to "it".

    Problem 2: Self Attention degenerates.
        - e_i . e_i = || e_i || ^2, which is usually a large positive number.
        - So, every word will attend to itself most strongly.
        - Everything else will be negligible in comparison.

    Problem 3: Each word plays a different role depending on who is looking at it.
        - Eg: When "it" is looking at "trophy", it needs to know that "trophy" is a singular noun, and can be big.
        - But when "big" is looking at "trophy", it needs to know that "trophy" is a noun that has some size.
        - A single vector (embedding), can't do this.

Following up to above question, why not use Key only then? Why do we need Value?


Q: Why do we need a different K and V? Why not just use K for both? 
A: - The key is concise repr that makes you findable. 
   - Value on the other hand is actual information to be extracted. 
   - Trying to read entire value to find the relevance would be computationally expensive.
   - Keys needs to be shaped to match queries well. 
   - Values needs to be shaped to be useful info for downstream layers.
 
---
# Cross Attention: 
Let's take a translation problem. 
English -> French
 "The cat didn't cross the street because it was too tired."

    - Let's say we're at postition 5, in french sentence.  Next words are not known. 
        ("Le chat n'a pas ..")
    - At this stage, the English has it's own self attention, French has it's own self attention. 
    - But, for next word in French, we need to look at English sentence as well. 
    - Here, the Q comes from French sentence and K,V comes from English sentence. 
    - French is decoder(Who needs information) side, English(Who has the information) is encoder side.


### Masking: 
    - In above case, decoder's future words are not know. In that case, the attention of words beyond (postion 5) are masked. 
    - But WHy? 
    - Reason 1: During training, if future words are not masked, model can cheat. 
    - Reason 2: Masking enables us to train effiiciently in parallel. 
        - Without masking, like RNN, we would say here are 4 words generated, predict 5. 
        - With maksing, we can send entire sentence, and mask ensures each postition can only see it's past. 
        - This way we can get N predictions in parallel with the cost of one forward pass.


# Multi-Head Attention:
- Every word requires to form a Q, so that it can know how to attend to other words. 
- But, a single Q may not be sufficient to capture all different aspects of relevance. 
- So, with multi-head attention, we have multiple sets of W_Q, W_K, W_V.
- Each set of W_Q, W_K, W_V is called a head.
- Each head learns to attend to different aspects of the sentence.
- For example, in the sentence "The cat didn't cross the street because it was too tired"
    - Take for "it", Questions it may ask are:  Who am I? What state am I in? What action am I involved in? There are 3 different aspects of relevance.
    - With multi-head attention, we can have 3 different heads, each learning to attend to different aspects of the sentence. 
    - One head may learn to attend to "cat", another head may learn to attend to "tired", and another head may learn to attend to "cross". 


"""
