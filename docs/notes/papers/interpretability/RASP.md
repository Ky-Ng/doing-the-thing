# Thinking Like Transformers
Authors: Gail Weiss, Yoav Goldberg, Eran Yahav

Publication Date: 2021-07-19

Full Paper: [Thinking Like Transformers](https://arxiv.org/pdf/2106.06981)

![rasp pipeline](assets/papers/rasp/RASP_pipeline.png)

## Introduction
The short name for this paper is known as RASP: `Restricted Access Sequence Processing Language`. 

1. We can think of RASP as a (functional) programming language for encoding computations that a transformer can learn at a high level. 

2. The RASP compiler can generate a transformer which implements the task (# of heads/layers auto-determined; note that width is not specified)

3. Train real transformer with additional loss penalty to resemble the attention matrices in the RASP transformer

## Relationship to MLP and Attention Computations

RASP code are constructed from functions which map sequences to sequences.

### MLP

- MLPs operate on a model element-wise. 

- RASP element-wise computations map sequence to sequence by executing the same computation on every element.

Examples:
```
indices("hi") 
-> [0,1]

tokens("hi")
-> ["h","i"]

(indices+1)("hi")
-> [1,2] because lambda (+ 1)[0,1] = [1,2]
```

### Attention

- Attention mechanism is broken into 3 sub-representations: `queries`, `keys`, `values`

- The `select` or `s-ops` emulate the `QK` circuit while the `aggregate` operation emulates the weighted sum of the `values`
        - one difference between the `select` matrix and the real attention matrices is that the output of a `select` is a bool mask
        - thus, `values` are always doing the average of the v vector representation (equivalent to `[0,0,1.0,0]` or `[0,0.5,0.5,0]` in an attention matrix)

Below is a walkthrough example of Figure 2
```
select([1,2,2],[0,1,2], ==) ~> q vector is [1,2,2] and v vector is [0,1,2]
  1 2 2
0 F F F
1 T F F
2 F T T 
```

- above we can see that select is like an attention heatmap

```
let the v vector be [4,6,8]

 (V)
F F F 
T F F 
F T T 

4 6 8
F F F 
T F F  
F T T

Row 1: 4(F) + 6(F) + 8(F) = 0
Row 2: 4(T) + 6(F) + 8(F) = 4
Row 3: 4(F) + 6(T) + 8(T) = (6+8) / 2 = 14/2 = 7
```