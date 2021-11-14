from keras.models import Sequential
import keras
from keras.optimizers import Adam
def create_grid_model(num_layers,n1,n2,n3,n4,n5,a1,a2,a3,a4,a5,aout,lr,b1,b2,ams):
    model = Sequential()
    model.add(keras.layers.Dense(n1, activation=a1 , input_dim=19))
    model.add(keras.layers.BatchNormalization())
    if num_layers == 2:
        model.add(keras.layers.Dense(n2, activation=a2))
    elif num_layers == 3:
        model.add(keras.layers.Dense(n2, activation=a2))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Dense(n3, activation=a3))
    elif num_layers == 4:
        model.add(keras.layers.Dense(n2, activation=a2))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Dense(n3, activation=a3))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Dense(n4, activation=a4))
    elif num_layers == 5:
        model.add(keras.layers.Dense(n2, activation=a2))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Dense(n3, activation=a3))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Dense(n4, activation=a4))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Dense(n5, activation=a5))
    model.add(keras.layers.Dense(1,activation=aout))
    opt = Adam(learning_rate=lr,beta_1=b1,beta_2=b2,amsgrad=ams)
    model.compile(loss='mean_squared_error', optimizer=opt)
    return model


def create_onehotencode_model2(input=19):
    model = Sequential()
    
    model.add(keras.layers.Dense(256, activation='relu' , input_dim=input))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dense(16, activation='elu'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dense(32, activation='elu'))
    model.add(keras.layers.Dense(1, activation='elu'))
    opt = Adam(learning_rate=0.005,amsgrad=True)
    model.compile(loss='mean_squared_error', optimizer=opt)

    return model


def create_onehotencode_model(input):
    model = Sequential()
    
    model.add(keras.layers.Dense(32, activation='relu' , input_dim=input))
    model.add(keras.layers.Dense(16, activation='relu'))
    model.add(keras.layers.Dense(1))
    
    model.compile(loss='mean_squared_error', optimizer='Adam')

    return model

def create_linear_model(input):
    model = Sequential()
    model.add(keras.layers.Dense(1 , input_dim=input))
    model.compile(loss='mean_squared_error', optimizer='SGD')

    return model

def create_embed_model(input_dim=4):
    model = Sequential()
    input_1 = keras.layers.Input((input_dim,), name="categorial_features_input")
    age_input = keras.layers.Input((1,),name="age_input")
    input_1_emb = keras.layers.Embedding(18, 15,  mask_zero=False)(input_1)
    input_1_emb = keras.layers.Flatten()(input_1_emb)

    
    outputs = keras.layers.Concatenate(axis=-1)([input_1_emb,age_input])

    outputs = keras.layers.Dense(256, activation='relu')(outputs)
    outputs = keras.layers.BatchNormalization()(outputs)
    outputs = keras.layers.Dense(16, activation='elu')(outputs)
    outputs = keras.layers.BatchNormalization()(outputs)
    outputs = keras.layers.Dense(32, activation='elu')(outputs)

    outputs = keras.layers.Dense(1, activation='elu')(outputs)
    model = keras.models.Model(inputs=[input_1, age_input], outputs=outputs)
    model.compile(loss='mean_squared_error', optimizer='adam')

    return model

def create_sep_embed_model():
    model = Sequential()
    input_1 = keras.layers.Input((1,))
    input_2 = keras.layers.Input((1,))
    input_3 = keras.layers.Input((1,))
    input_4 = keras.layers.Input((1,))
    #input_5 = keras.layers.Input((1,))
    age_input = keras.layers.Input((1,),name="age_input")

    input_1_emb = keras.layers.Embedding(3, 2,  mask_zero=False)(input_1)
    input_1_emb = keras.layers.Flatten()(input_1_emb)

    input_2_emb = keras.layers.Embedding(4, 2,  mask_zero=False)(input_2)
    input_2_emb = keras.layers.Flatten()(input_2_emb)

    input_3_emb = keras.layers.Embedding(5, 3,  mask_zero=False)(input_3)
    input_3_emb = keras.layers.Flatten()(input_3_emb)

    input_4_emb = keras.layers.Embedding(3, 2,  mask_zero=False)(input_4)
    input_4_emb = keras.layers.Flatten()(input_4_emb)

    #input_5_emb = keras.layers.Embedding(3, 2,  mask_zero=False)(input_5)
    #input_5_emb = keras.layers.Flatten()(input_5_emb)

    
    outputs = keras.layers.Concatenate(axis=-1)([input_1_emb,input_2_emb,input_3_emb,input_4_emb,#input_5_emb,
    age_input])

    outputs = keras.layers.Dense(256, activation='relu')(outputs)
    outputs = keras.layers.BatchNormalization()(outputs)
    outputs = keras.layers.Dense(16, activation='elu')(outputs)
    outputs = keras.layers.BatchNormalization()(outputs)
    outputs = keras.layers.Dense(32, activation='elu')(outputs)

    outputs = keras.layers.Dense(1, activation='elu')(outputs)
    model = keras.models.Model(inputs=[input_1,input_2,input_3,input_4,#input_5,
    age_input], outputs=outputs)
    model.compile(loss='mean_squared_error', optimizer='adam')

    return model

def create_sep_embed_model_leave_one():
    model = Sequential()
    input_1 = keras.layers.Input((1,))
    input_2 = keras.layers.Input((1,))
    input_3 = keras.layers.Input((1,))
    #input_4 = keras.layers.Input((1,))
    #input_5 = keras.layers.Input((1,))
    age_input = keras.layers.Input((1,),name="age_input")

    input_1_emb = keras.layers.Embedding(3, 2,  mask_zero=False)(input_1)
    input_1_emb = keras.layers.Flatten()(input_1_emb)

    input_2_emb = keras.layers.Embedding(4, 2,  mask_zero=False)(input_2)
    input_2_emb = keras.layers.Flatten()(input_2_emb)

    input_3_emb = keras.layers.Embedding(5, 3,  mask_zero=False)(input_3)
    input_3_emb = keras.layers.Flatten()(input_3_emb)
    #input_5_emb = keras.layers.Embedding(3, 2,  mask_zero=False)(input_5)
    #input_5_emb = keras.layers.Flatten()(input_5_emb)

    
    outputs = keras.layers.Concatenate(axis=-1)([input_1_emb,input_2_emb,input_3_emb,#input_5_emb,
    age_input])

    outputs = keras.layers.Dense(256, activation='relu')(outputs)
    outputs = keras.layers.BatchNormalization()(outputs)
    outputs = keras.layers.Dense(16, activation='elu')(outputs)
    outputs = keras.layers.BatchNormalization()(outputs)
    outputs = keras.layers.Dense(32, activation='elu')(outputs)

    outputs = keras.layers.Dense(1, activation='elu')(outputs)
    model = keras.models.Model(inputs=[input_1,input_2,input_3,#input_5,
    age_input], outputs=outputs)
    model.compile(loss='mean_squared_error', optimizer='adam')

    return model