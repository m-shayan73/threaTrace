## Update on the use of test data during training phase for DARPA datasets

The author acknowledged the issue in an email reply.

![](./assets/test_data_use_reply.png)

Hence, I commented out the validation code (in which the models are being validated using test data and if they performed poorly are ebing deleted). Then, I trained and evaluated the system again on all DARPA datasets. The results were poor and are noted below:

1) E3 Cadets:

    <pre>
    tp: 6
    fp: 7072
    tn: 337250
    fn: 12846
    precision: 0.0008476970895733138
    recall: 0.000466853408029875
    fscore: 0.0006021073300120611
    </pre>

2) E3 Fivedirections:

    <pre>
    tp: 35
    fp: 100
    tn: 373959
    fn: 390
    precision: 0.25925925925906723
    recall: 0.08235294117645121
    fscore: 0.1249999996336417
    </pre>

3) E3 Theia:

    <pre>
    tp: 4508
    fp: 6505
    tn: 312941
    fn: 20811
    precision: 0.4093344229546862
    recall: 0.1780481061653297
    fscore: 0.24815589558694834
    </pre>

5) E3 Trace:

    <pre>
    tp: 23
    fp: 66
    tn: 1143999
    fn: 67360
    precision: 0.2584269662918445
    recall: 0.0003413323835388743
    fscore: 0.0006817642871446428
    </pre>
