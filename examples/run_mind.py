import numpy as np

from easymatch.models import MIND
from deepctr.inputs import SparseFeat, VarLenSparseFeat, DenseFeat, get_feature_names
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.models import Model

def get_xy_fd():
    feature_columns = [SparseFeat('user', 3, embedding_dim=10), SparseFeat(
        'gender', 2, embedding_dim=4), SparseFeat('item', 3 + 1, embedding_dim=8),
                       SparseFeat('item_gender', 2 + 1, embedding_dim=4), DenseFeat('score', 1)]
    feature_columns += [
        VarLenSparseFeat(SparseFeat('hist_item', vocabulary_size=3 + 1, embedding_dim=20, embedding_name='item'),
                         maxlen=4),
        VarLenSparseFeat(SparseFeat('hist_item_gender', 2 + 1, embedding_dim=4, embedding_name='item_gender'),
                         maxlen=4)]

    feature_columns += [DenseFeat('hist_len', 1, dtype="int64")]

    behavior_feature_list = ["item"]
    uid = np.array([0, 1, 2])
    ugender = np.array([0, 1, 0])
    iid = np.array([1, 2, 3])  # 0 is mask value
    igender = np.array([1, 2, 1])  # 0 is mask value
    score = np.array([0.1, 0.2, 0.3])

    hist_iid = np.array([[1, 2, 3, 0], [1, 2, 3, 0], [1, 2, 0, 0]])
    hist_igender = np.array([[1, 1, 2, 0], [2, 1, 1, 0], [2, 1, 0, 0]])
    hist_len = np.array([3, 3, 2])

    feature_dict = {'user': uid, 'gender': ugender, 'item': iid, 'item_gender': igender,
                    'hist_item': hist_iid, 'hist_item_gender': hist_igender, 'hist_len': hist_len, 'score': score}
    x = {name: feature_dict[name] for name in get_feature_names(feature_columns)}
    y = [1, 1, 1]
    return x, y, feature_columns, behavior_feature_list


def custom_loss(y_true, y_pred):
    return K.mean(y_pred)


if __name__ == "__main__":
    x, y, feature_columns, behavior_feature_list = get_xy_fd()
    model = MIND(feature_columns, behavior_feature_list, 3 + 1)
    model.compile('adam', loss=custom_loss)
    history = model.fit(x, y, verbose=1, epochs=10, validation_split=0.5)
    user_embedding_model = Model(inputs=model.input, outputs=model.get_layer("user_embedding").output)
    user_embedding = user_embedding_model.predict(x)
    item_embedding = model.get_layer("sampled_softmax_layer").get_weights()[0]
    print("user_embedding", user_embedding)
    print("item_embedding", item_embedding)
