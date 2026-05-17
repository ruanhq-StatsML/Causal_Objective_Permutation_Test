#autotst:
from autogluon.tabular import TabularPredictor, TabularDataset
from autogluon.vision import ImagePredictor, ImageDataset
import pandas as pd
import numpy as np
import warnings
#from .autotst_types import Dataset, Weights, Predictions
import typing
import numpy as np
import nptyping as npt
from nptyping import NDArray, Shape, Float64, UInt
ListFloats = NDArray[Shape["1"], Float64]
Weights = ListFloats
Predictions = ListFloats
Labels = NDArray[Shape["1"], UInt]
Samples = NDArray[typing.Any, typing.Any]
Dataset = NDArray[typing.Any, typing.Any]

class SplittedSets:
    """
    Class encapsulating datasets and labels dividing into testing and training.
    """
    def __init__(
        self,
        training_set: Dataset,
        test_set: Dataset,
        training_labels: Labels,
        test_labels: Labels,
    ):
        self.training_set = training_set
        self.test_set = test_set
        self.training_labels = training_labels
        self.test_labels = test_labels
    def training_split(self):
        """
        Returns the number p and q of items that have been drawn
        respectively from the distributions P and Q
        in the training set. The first pth items of the trainign set
        correspond to P, and the following qth items correspond to Q.
        """
        p = len(np.where(self.training_labels == 1)[0])
        q = len(self.training_set) - p
        return p, q
    def test_split(self):
        """
        Similar to training_split, but for the testing set.
        """
        p = len(np.where(self.test_labels == 1)[0])
        q = len(self.test_set) - p
        return p, q
    @staticmethod
    def split(
        X: Samples, Y: Samples, split_ratio: float
    ) -> typing.Tuple[Dataset, Dataset, Labels, Labels]:
        """
        Creates a labeled dataset that concatenates the samples drawn from the distributions
        X and Y, and splits it between a training and a testing sets. Labels are binaries with
        values 1 for samples drawn from P and 0 for samples drawn from Q.
        The returned tuples has for values: training set, testing set, labels for training set,
        labels for testing set.
        """
        if type(X) != list and X.shape[1:] != Y.shape[1:]:
            raise ValueError("X and Y should be samples of items of same dimension")
        if split_ratio < 0 or split_ratio > 1:
            raise ValueError("split ratio should be between 0 and 1")
        n = len(X)
        n_train = int(n * split_ratio)
        m = len(Y)
        m_train = int(m * split_ratio)
        X_train, X_test = X[:n_train], X[n_train:]
        Y_train, Y_test = Y[:m_train], Y[m_train:]
        data_train = np.concatenate((X_train, Y_train))
        data_test = np.concatenate((X_test, Y_test))
        label_train = np.array([1] * n_train + [0] * m_train)
        label_test = np.array([1] * (n - n_train) + [0] * (m - m_train))
        return data_train, data_test, label_train, label_test
    @classmethod
    def from_samples(
        cls, sample_p: Samples, sample_q: Samples, split_ratio: float = 0.5
    ) -> object:
        """
        Creates a labeled dataset that concatenates the samples drawn from the distributions
        P and Q, and splits it between a training and a testing sets. Labels are binaries with
        values 1 for samples drawn from P and 0 for samples drawn from Q.
        """
        return cls(*cls.split(sample_p, sample_q, split_ratio))

class Model:
    """
    Generic model class for two-sample tests
    """
    def __init__(self, **kwargs):
        raise NotImplementedError()
    def fit(self, data_train, label_train, weights, **kwargs):
        raise NotImplementedError()
    def predict(self, data_test):
        raise NotImplementedError()


class AutoGluonTabularPredictor(Model):
    """
    Wrapper model for the Tabular Predictor of the AutoGluon
    package
    """

    def __init__(self, **kwargs) -> None:
        self.model = TabularPredictor(
            label="label", sample_weight="weights", problem_type="regression", **kwargs
        )

    def fit(
        self,
        data_train: Dataset,
        label_train: Dataset,
        weights: Weights,
        presets: str = "best_quality",
        time_limit: int = 60,
        verbosity: int = 0,
        **kwargs
    ) -> None:
        """
        Wrapper around fit routine.
        :param data_train: training data
        :param label_train: training labels
        :param weights: weights for the loss
        :param presets: Autogluon preset
        :param time_limit: time limit for train (seconds)
        :param verbosity: control output of Autogluon
        :param kwargs: other arguments to be passed to AutoGluon's fit routine.
        :return:
        """
        df_train = pd.DataFrame(data_train)
        df_train["label"] = label_train
        df_train["weights"] = weights
        df_train = TabularDataset(df_train)
        self.model.fit(
            df_train,
            presets=presets,
            time_limit=time_limit,
            verbosity=verbosity,
            **kwargs
        )

    def predict(self, data_test: Dataset) -> Predictions:
        df_test = TabularDataset(pd.DataFrame(data_test))
        return self.model.predict(df_test)


class AutoGluonImagePredictor(Model):
    """
    Wrapper model for the Image Classifier of the AutoGluon
    package.
    The objective is classification, and the witness function uses the predicted probabilities.
    """
    def __init__(self, **kwargs) -> None:
        self.model = ImagePredictor(label="label", verbosity=0, **kwargs)
    def fit(
        self,
        data_train: Dataset,
        label_train: Dataset,
        weights: Weights,
        presets: str = "best_quality",
        time_limit: int = 60,
        **kwargs
    ) -> None:
        """
        Wrapper around fit routine.
        :param data_train: training data - provided as a list of image paths!
        :param label_train: training labels
        :param weights: weights for the loss - will be ignored here!!!
        :param presets: Autogluon preset
        :param time_limit: time limit for train (seconds)
        :param kwargs: other arguments to be passed to AutoGluon's fit routine.
        :return:
        """
        if weights[0] != 0.5:
            warnings.warn(
                "AutoGluonImagePredictor ignores the weights! consider oversampling or using another model."
            )
        df_train = pd.DataFrame({"image": data_train, "label": label_train})
        self.model.fit(df_train, presets=presets, time_limit=time_limit, **kwargs)
    def predict(self, data_test: Dataset) -> Predictions:
        df_test = pd.DataFrame({"image": data_test})
        predictions = np.array(self.model.predict_proba(df_test))
        predictions = predictions[:, 1]  # return probability of class '1'
        return predictions


def permutations_p_value(
    predictions: Predictions, labels: Labels, permutations: int = 10000
) -> float:
    """
    Compute p value of the witness mean discrepancy test statistic via permutations

    :param predictions: one-dimensional array with the witness predictions of the test data
    :param labels: one-dimensional array with labels 1 and 0 indicating data coming from P or Q
    :param int permutations: Number of permutations
    :return: p value
    """

    if len(predictions.shape) > 1:
        raise ValueError("predictions should be one dimentional")

    if len(labels.shape) > 1:
        raise ValueError("labels should be one dimentional")

    if len(predictions) != len(labels):
        raise ValueError("predictions and labels should be of the same length")

    p_samp = predictions[labels == 1]
    q_samp = predictions[labels == 0]
    tau = np.mean(p_samp) - np.mean(q_samp)  # value on original partition
    p = 1 / (permutations + 1)
    for i in range(0, permutations):
        np.random.shuffle(predictions)
        p_samp = predictions[labels == 1]
        q_samp = predictions[labels == 0]
        tau_sim = np.mean(p_samp) - np.mean(q_samp)

        if tau <= tau_sim:
            p += 1.0 / (permutations + 1)

    return p


def get_weights(label_train: Labels) -> Weights:
    """
    Labels being a one-dimensional array with labels 1 and 0, returns an array of
    weights that gives higher values to indexes corresponding to the less represented
    label.
    """

    n1 = len(np.where(label_train)[0])
    n2 = len(label_train) - n1

    ratio = n1 / (n1 + n2)
    return np.array([1.0 - ratio] * n1 + [ratio] * n2)


def fit_witness(
    data_train: Dataset, label_train: Dataset, model: Model, **kwargs
) -> None:
    """
    Calls the fit function of the model on the provided dataset, weighted to account
    for the difference of representation of the two labels.
    :param predictions: one-dimensional array with the witness predictions of the test data
    :param labels: one-dimensional array with labels 1 and 0 indicating data coming from one sample or the other
    :param model: the model on which the fit function is applied.
    """

    if len(data_train) != len(label_train):
        raise ValueError("data_train and label_train should be of the same length")

    weights = get_weights(label_train)
    model.fit(data_train, label_train, weights, **kwargs)


def p_value_evaluate(
    model: Model, data_test: Dataset, labels_test: Labels, permutations: int = 10000
) -> typing.Tuple[Dataset, float]:

    """
    Apply the model to generate predictions, and uses these predictions to evaluate the  p value.
    :param model: the model used for prediction, assumed to have been fitted
    :param dataset: dataset
    :param labels: one-dimensional array with labels 1 and 0 indicating data coming from one sample or the other
    :param permutations: number of permutations when estimating the p-value
    :return: the predictions and the p value
    """

    prediction_test = np.array(model.predict(data_test))
    return (
        prediction_test,
        permutations_p_value(prediction_test, labels_test, permutations),
    )


def get_default_model() -> Model:
    """
    Returns an instance of the AutoGluonTabularPredictor, with default parameters
    """
    return AutoGluonTabularPredictor()


def p_value(
    sample_p: Samples,
    sample_q: Samples,
    model: typing.Optional[Model] = None,
    split_ratio: float = 0.5,
    permutations: int = 10000,
    **fit_kwargs
) -> float:

    """
    Split the datasets unto a training and a test set, fit the model using the training set
    and uses the test set to compute the p-value.
    :param sample_p: samples drawn from a first distribution
    :param sample_q: samples drawn from a second distribution
    :param model: instance of model for fitting and prediction. If None (the default): an AutoGluonTabularPredictor will be used
    :param split_ratio: for splitting into learning and testing sets
    :param permutations: number of permutations used to estimate the p value
    :param fit_kwargs: parameters to the model's fit function
    :return: p value
    """

    if model is None:
        model = get_default_model()

    splitted_sets = typing.cast(
        SplittedSets, SplittedSets.from_samples(sample_p, sample_q, split_ratio)
    )

    fit_witness(
        splitted_sets.training_set, splitted_sets.training_labels, model, **fit_kwargs
    )

    return p_value_evaluate(
        model,
        splitted_sets.test_set,
        splitted_sets.test_labels,
        permutations=permutations,
    )[1]


def interpret(
    data_test: Dataset, predictions: Predictions, k: int = 1
) -> typing.Tuple[Dataset, Dataset]:

    """
    Returns the k most typical examples from the two distributions
    :param data_test: dataset with the first items corresponding to the first distribution and the last items to the second distributions
    :param predictions: label prediction corresponding to the dataset
    :param k: number of items to extract from the dataset, for each distribution
    :return: the k most typical examples from the two distributions
    """

    if len(data_test) != len(predictions):
        raise ValueError("data_test and predictions should be of the same length")

    most_typical = np.argsort(predictions)
    p_typical = data_test[most_typical[:+k]]
    q_typical = data_test[most_typical[-k:]]
    return p_typical, q_typical





class AutoTST:
    """
    AutoML Two-Sample Test

    Documentation with example of the class goes here
    """

    def __init__(
        self,
        sample_p: Samples,
        sample_q: Samples,
        split_ratio: float = 0.5,
        model: typing.Type[Model] = AutoGluonTabularPredictor,
        **model_kwargs
    ) -> None:
        """
        Constructor

        :param sample_p: Sample drawn from P
        :param sample_q: Sample drawn from Q
        :param split_ratio: Ratio that defines how much data is used for training the witness
        :param model: Model used to learn the witness function
        :param **model_kwargs: Keyword arguments to initialize the model
        :return: None
        """
        self.X = sample_p
        self.Y = sample_q
        self.model = model(**model_kwargs)
        self.split_ratio = split_ratio
        self.size_ratio = len(sample_p) / (len(sample_p) + len(sample_q))
        self.splitted_sets: typing.Optional[SplittedSets] = None
        self.prediction_test: typing.Optional[Predictions] = None
        self._fitted = False

    def split_data(self) -> SplittedSets:
        """
        Split & label data using the instances splitting ratio. The splits are stored as attributes but also returned.
        """
        self.splitted_sets = typing.cast(
            SplittedSets, SplittedSets.from_samples(self.X, self.Y, self.split_ratio)
        )
        return self.splitted_sets

    def fit_witness(self, **kwargs) -> None:
        """
        Fit witness

        :param kwargs: Keyword arguments to be passed to fit method of model
        :return: None
        """
        if self.splitted_sets is None:
            raise ValueError("split_data should be called first")
        data_train = self.splitted_sets.training_set
        label_train = self.splitted_sets.training_labels
        functions.fit_witness(data_train, label_train, self.model, **kwargs)
        self._fitted = True

    def p_value_evaluate(self, permutations: int = 10000) -> float:
        """
        Evaluate p value.

        :param permutations: number of permutations when estimating the p-value
        :return: p value
        """
        if permutations < 0:
            raise ValueError("permutations should be positive")
        if not self._fitted:
            raise ValueError("the model should be trained first")
        if not self.splitted_sets:
            raise ValueError("split_data should be called first")
        data_test = self.splitted_sets.test_set
        label_test = self.splitted_sets.test_labels
        self.prediction_test = np.array(self.model.predict(data_test))
        return functions.permutations_p_value(
            self.prediction_test, label_test, permutations=permutations
        )

    def p_value(self, permutations: int = 10000, **fit_kwargs):
        """
        Run the complete pipeline and return p value with default settings.

        :return: p-value
        """
        self.split_data()
        self.fit_witness(**fit_kwargs)
        pval = self.p_value_evaluate(permutations=permutations)
        return pval

    def interpret(self, k=1):
        """
        Return the k most typical examples from P and Q.

        :return: Tuple: (k most significant examples from P, k most significant examples from Q)
        """
        if self.prediction_test is None:
            raise RuntimeError(
                "Interpretation can only be done after the p-value was computed."
            )
        p, q = self.splitted_sets.training_split()
        if k > p or k > q:
            raise ValueError("k should be between {} and {}".format(p, q))
        return functions.interpret(self.splitted_sets.test_set, self.prediction_test, k)


