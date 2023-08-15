num_workers = 1
target_domain = 'movie'
history_size = 5

# DIN
class DinConfig():
    train_batch_size = 1024
    test_batch_size = 10000
    embedding_size = 128

    device = 'cuda:0'
    # device = 'cpu'
    learning_rate = 0.001
    start_epoch = 0
    epochs = 100
    interval = 300
    save_epochs = 0
    description = "Din"

    load_path = './log/amazon/log1/run'
    csv_path = load_path[0:-3] + 'din'+target_domain+'.csv'
    checkpoint_path = load_path[0:-3] + 'checkpoints/din_'

class Iv4RecConfig():
    train_batch_size = 1024
    test_batch_size = 10000
    embedding_size = 128

    device = 'cuda:0'
    # device = 'cpu'
    learning_rate = 0.0003
    start_epoch = 0
    epochs = 100
    interval = 300
    load_path = './log/amazon/log2/run'
    csv_path = load_path[0:-3] + 'ivdin_'+target_domain+'.csv'
    checkpoint_path = load_path[0:-3] + 'checkpoints/iv4rec_'
    description = 'Iv4Rec'
    mlp = {'hidden_dims': [embedding_size,embedding_size],'dropout': [0.1],'is_dropout': True,'activation': 'tanh'}
    alpha = {'hidden_dims': [64*2,128,32,1],'dropout': [], 'is_dropout': False,'activation': 'tanh'}
    beta = alpha
    l2_lambda = 1e-6
    connection_type = "concat"

class ArisenConfig():
    train_batch_size = 1024
    test_batch_size = 10000
    embedding_size = 128

    device = 'cuda:0'
    # device = 'cpu'
    learning_rate = 0.0001
    start_epoch = 0
    epochs = 100
    interval = 300
    load_path = './log/amazon/log3/run'
    csv_path = load_path[0:-3] + 'arisen_'+target_domain+'.csv'
    checkpoint_path = load_path[0:-3] + 'checkpoints/arisen'
    description = 'Arisen'
    mlp = {'hidden_dims': [embedding_size,embedding_size],'dropout': [],'is_dropout': False,'activation': 'tanh'}
    alpha = {'hidden_dims': [embedding_size*2,256,64,16,1],'dropout': [], 'is_dropout': False,'activation': 'relu'}
    beta = alpha
    l2_lambda = 1e-5

    mine_dropout = 0.1
    mi_weight = 0.1
    mi_loss= True

    '''
    connection_type:
    '''
    # connection_type = "concat"
    connection_type = "z"
    z_weight = 0.1

class SequenceModelConfig():
    train_batch_size = 1024
    test_batch_size = 10000
    embedding_size = 128

    device = 'cuda:0'
    learning_rate = 0.0001
    start_epoch =0
    epochs = 100
    interval = 300
    save_epochs = 0
    description = 'SequenceModel'
    seq_type = "GRU"
    # seq_type = "Transformer"

    load_path = './log/amazon/log4/run'
    csv_path = load_path[0:-3] + seq_type +'_'+target_domain+ '.csv'
    checkpoint_path = load_path[0:-3] + 'checkpoints/' + seq_type + '_'

class Arisen_SequenceModelConfig():
    train_batch_size = 1024
    test_batch_size = 10000
    embedding_size = 128

    device = 'cuda:0'
    # device = 'cpu'
    learning_rate = 0.0001
    start_epoch = 0
    epochs = 100
    interval = 100
    save_epochs = 0
    description = 'Arisen_SequenceModel'
    seq_type = "GRU"
    # seq_type = "Transformer"

    load_path = './log/amazon/log5/run'
    csv_path = load_path[0:-3] + seq_type +'_arisen_'+target_domain + '.csv'
    checkpoint_path = load_path[0:-3] + 'checkpoints/' + seq_type + '_'

    mlp = {'hidden_dims': [embedding_size, embedding_size], 'dropout': [], 'is_dropout': False, 'activation': 'tanh'}
    alpha = {'hidden_dims': [embedding_size * 2, 64, 16, 32, 1], 'dropout': [], 'is_dropout': False,'activation': 'tanh'}
    beta = alpha
    l2_lambda = 1e-5

    mine_dropout = 0.1
    mi_weight = 0.01
    mi_loss = True

    '''
    connection_type:
    '''
    # connection_type = "concat"
    connection_type = "z"
    z_weight = 0.1

class CaserConfig():
    train_batch_size = 1024
    test_batch_size = 10000
    embedding_size = 128

    device = 'cuda:0'
    # device = 'cpu'
    learning_rate = 0.001
    start_epoch =0
    epochs = 100
    interval = 100
    save_epochs = 0
    description = "Caser"

    load_path = './log/amazon/log6/run'
    csv_path = load_path[0:-3] + 'caser.csv'
    checkpoint_path = load_path[0:-3] + 'checkpoints/caser_'

class Arisen_CaserConfig():
    train_batch_size = 1024
    test_batch_size = 10000
    embedding_size = 128

    device = 'cuda:0'
    learning_rate = 0.001
    start_epoch =0
    epochs = 100
    interval = 100
    save_epochs = 0
    description = "Arisen_Caser"

    load_path = './log/amazon/log7/run'
    csv_path = load_path[0:-3] + 'arisen_caser.csv'
    checkpoint_path = load_path[0:-3] + 'checkpoints/arisen_caser_'

    mlp = {'hidden_dims': [embedding_size, embedding_size], 'dropout': [0.1], 'is_dropout': True, 'activation': 'tanh'}
    alpha = {'hidden_dims': [embedding_size * 2, 512, 128, 32, 1], 'dropout': [], 'is_dropout': False,'activation': 'tanh'}
    beta = alpha
    l2_lambda = 1e-5
    mine_dropout = 0.1
    mi_weight = 0.1
    mi_loss = True
    connection_type = None