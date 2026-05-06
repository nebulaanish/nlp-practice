"""
Regular RNNs can't remember long-term dependencies.
GRU tries to solve this problem by introducing gating mechanisms that control the flow of information.
LSTM (Long Short-Term Memory) is another type of RNN that also addresses the vanishing gradient problem.

- Cell State: Controls the flow of information through the network.
- Cell states are managed by using 3 gates: the forget gate, the input gate, and the output gate.
- As compared to reset and update gates in GRU.
- the update gate in GRU is similar to forget gate + input gate in LSTM. So GRU is faster. 

Gate1: Forget Gate:
- Decides what information to throw away from the cell state.

Gate2: Input Gate:
- What new information is worth adding to the cell state.

Gate3: Output Gate:
- What to output right now based on the cell state.


Mathematical Intuition: 
Let C_{t-1} be long term memory from previous step 
    h_{t-1 be immediate context, that LSTM just output at previous step. 
    e_t be current input, i.e word embedding of current word.

Step 1: Forget Gate
- LSTM decides what information to throw away.
- It looks at Short-Term memory(h_{t-1}) and new input (e_t), and 
- Outputs a number between 0 and 1 for each number in the cell state C_{t-1}. 
- A 1 represents "completely keep this" while a 0 represents "completely get rid of this".

Step 2: Input Gate (Candidate Memory) 
- LSTM needs to generate the new information that I might want to add to the cell state.
- Looks at h_{t-1} and e_t, and outputs a vector of new candidate values, C~_t, that could be added to the state.
- Uses tanh to push the values to be between -1 and 1.

Step 3: Update Cell State
- LSTM shouldn't just dump all candidate values into the cell state. 
- So, it creates an input score (0-1). 
- 0 means don't write this candidate to cell state, 1, means this candidate is super important. 
- candiate values are scaled by this input score, and then added to the cell state.

Step 4: Actual Output  
- LSTM needs to decide what to output.
- C_t = forget_gate * C_{t-1} + input_gate * C~_t
- This only updates the cell state. 
- The new h_t is calculated as follows:
- First, we run the cell state through tanh to push the values to be between -1 and 1. 
- Then, we multiply it by the output of the output gate, which decides what parts of the cell state to output.

"""
