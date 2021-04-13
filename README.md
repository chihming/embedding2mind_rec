# embedding2mind_rec

Suppose we have the embedding file **hpe.embed**:
```
U87243 -0.406477 -0.00270414 -0.382542 0.347666
U532401 -0.153042 -0.263854 0.274493 0.134442
U239687 0.13836 0.208176 0.121421 -0.331204
...
```
and the given behavior file **behaviors.tsv**:
```
...
29      U32943  11/15/2019 1:10:32 PM   N43371 N23421 N81598 N81289 N82405 N112333 N5952 N75132 N122773 N3785 N82809 N79062 N87946 N31079 N122006 N73137 N115296 N88189 N119999-0 N46555-0 N2110-0 N21018-0 N58760-0 N129416-0 N29160-0 N112536-0 N46641-0 N89764-1 N91737-0
30      U700617 11/15/2019 12:49:44 PM  N61504 N29947 N98238 N118038 N62427 N49594 N25951 N7237 N128031 N21018-1 N46555-0 N129416-0 N2110-0
31      U477175 11/15/2019 1:27:59 PM   N95287 N7154 N43428 N44660 N72579 N126495 N71258        N117802-0 N91737-0 N56602-0 N89764-0 N44294-0 N52464-1 N100425-0 N119999-0 N30206-0 N47891-0 N49307-0 N29160-0 N91268-0 N58760-0 N129416-0 N39770-0 N112536-0 N92905-0 N21018-0 N40742-0 N2110-0 N46555-0 N46641-0 N61442-0 N99846-0
...
```
This command helps generate the recommendations for mind-news task.
```python
python3 embedding2mind_rec/predict.py --query_embed ./exp/hpe.embed --target_embed ./exp/hpe.embed --behavior_file behaviors.tsv > prediction.txt
```
You'll get a **predicito.txt** file likes:
```
...
11 [16,3,23,5,9,7,14,22,11,19,10,2,17,25,8,21,18,20,15,6,13,24,1,4,12]
12 [2,3,4,1]
13 [7,9,2,13,3,10,11,12,5,6,8,14,1,4]
...
```
which can be evaluated by https://github.com/msnews/MIND/blob/master/evaluate.py.

HOWEVER, their released evaluation code is not easy-to-use so that you can try my modified version by:
```
python3 embedding2mind_rec/evaluate.py prediction.txt
```
which evaluates the predictions based on the mind-news **dev** data, and outputs the results like:
```
AUC:0.6523
MRR:0.3145
nDCG@5:0.3414
nDCG@10:0.4046
```
