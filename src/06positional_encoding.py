"""

# Positional Encoding:
    - Attention mechanism is permutation invariant. It is same for a given set of words.
    Eg: "dog bites man" and "man bites dog" would have same attention scores. ? Why? Worth tracing the formula on this example.
    - To overcome this, we can add the position to the embedding. i.e e_i now becomes e_i + p_i, where p_i is the positional encoding for position i.

    Approaches:
        1. Naive:  We concatenate increasing order of numbers to embeddings.
            eg: e_i + 1, e_(i+1) + 2, e_(i+2) + 3, ...
            Problem: Numbers like these can become large easily and can skew the attention scores. eg e_i + 10000.
                - Additionally, models doesn't have intuition of these discrete numbers like humans do.
                - So we can't use discrete, unbounded values.
                - We want continuous, bounded values that provide ordering or nuance of distance too.
        2. Sine Function:
            - They provide boudned, continuous values. Eg, sine(1), sine(2), sine(3), ...
            - But, since they are periodic, there can be numbers such that sine(x) = sine(y), which can cause confusion. So uniqueness of position can't be maintained.

        3. Two vector (Sine and Cosine):
            - We can use two vectors for positional encoding.
            - eg: p_i = [sin(i), cos(i)]
            - This still doesn't ensure uniqueness. The [sin(i), cos(i)] pair repeats periodically.

        4. Four vectors [sin, cosine, sin(i/2), cos(i/2)] i.e with different frequencies:
            - This further reduces the likelihood of repetition. i/2 is better than 2i for this case too.
            - If 4 vectors can't ensure uniquess?
            - What if we use a long enough sequence such that, a unique positional encoding is generated for sufficiently long sequence?
            - [sin(i), cos(i), sin(i/2), cos(i/2), sin(i/3), cos(i/3), ...] with different frequencies.
            - 512 dimensional positional encoding

        Mathematical Derivation of Approach 4:
            - We have sin in even index and cos in odd index.
            - PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
            - PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
            Where,
                - i: index of the dimension
                - pos: position of the word in the sentence
                - d_model: dimension of the model (embedding size): 512 in original transformer paper.

        - The authors hypothesize that, if we know PE_(pos), we can predic the PE_(pos+k) for small k. 
            i.e there exists a T_k matrix, such that PE_(pos+k) = T_k * PE_(pos)

            
    Q: But why addition and not concatenation? With concatenation, we can have separate parameters for position and word.
    A:

    Q: What should p_i look like?
    A:

    Q: What kind of position should the model know? Absolute or relative? or both? What kind of patterns in language are position-dependent.
    A:
"""
