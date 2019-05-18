testsets = ['dev.txt','twitter-test1.txt', 'twitter-test2.txt', 'twitter-test3.txt']
# testsets = ['dev.txt']
# testsets = ['new-dev.txt','twitter-test1.txt', 'twitter-test2.txt', 'twitter-test3.txt']
# testsets = ['twitter-dev-data.txt'] # You may uncomment this line while you are experimenting with your classifier

# def read_training_data(training_data):
#     id_gts = {}
#     with open(training_data, 'r', encoding='utf-8') as f:
#         for line in f:
#             fields = line.split('\t')
#             tweetid = fields[0]
#             gt = fields[1]
#             content = fields[2].strip()
#             id_gts[tweetid] = gt, content
#     print(id_gts)
#     return id_gts
#
# for testset in testsets:
#     # TODO: classify tweets in test set
#     # if testset == 'twitter-test1.txt':
#     test = read_training_data(testset)