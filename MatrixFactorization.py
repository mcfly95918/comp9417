import numpy as np
from six.moves import range


class InvalidPrediction(Exception):
    """Exception raised when a prediction is impossible.

    When raised, the estimation :math:`\hat{r}_{ui}` is set to the global mean
    of all ratings :math:`\mu`.
    """
    pass


class MatrixFactorization:
    """
    The prediction :math:`\\hat{r}_{ui}` is set as:
    .. math::
        \hat{r}_{ui} = \mu + b_u + b_i + q_i^Tp_u
    If user :math:`u` is unknown, then the bias :math:`b_u` and the factors
    :math:`p_u` are assumed to be zero. The same applies for item :math:`i`
    with :math:`b_i` and :math:`q_i`.
    
    To estimate all the unknown, we minimize the following regularized squared
    error:
    .. math::
        \sum_{r_{ui} \in R_{train}} \left(r_{ui} - \hat{r}_{ui} \\right)^2 +
        \lambda\\left(b_i^2 + b_u^2 + ||q_i||^2 + ||p_u||^2\\right)
    The minimization is performed by a very straightforward stochastic gradient
    descent:
    .. math::
        b_u &\leftarrow b_u &+ \gamma (e_{ui} - \lambda b_u)
        b_i &\leftarrow b_i &+ \gamma (e_{ui} - \lambda b_i)
        p_u &\leftarrow p_u &+ \gamma (e_{ui} \cdot q_i - \lambda p_u)
        q_i &\leftarrow q_i &+ \gamma (e_{ui} \cdot p_u - \lambda q_i)
    where :math:`e_{ui} = r_{ui} - \hat{r}_{ui}`. These steps are performed
    over all the ratings of the trainset and repeated ``n_epochs`` times.
    Biases are initialized to ``0``. User and item factors are randomly
    initialized according to a normal distribution, which can be tuned using
    the ``init_mean`` and ``init_std_dev`` parameters.
    You also have control over the learning rate :math:`\gamma` and the
    regularization term :math:`\lambda`. Both can be different for each
    kind of parameter (see below). By default, learning rates are set to
    ``0.005`` and regularization terms are set to ``0.02``.
    .. _unbiased_note:
    .. note::
        You can choose to use an unbiased version of this algorithm, simply
        predicting:
        .. math::
            \hat{r}_{ui} = q_i^Tp_u
        This is equivalent to Probabilistic Matrix Factorization and can be 
        achieved by setting the ``biased`` parameter to ``False``.
    Args:
        n_factors: The number of factors. Default is ``100``.
        n_epochs: The number of iteration of the SGD procedure. Default is
            ``20``.
        biased(bool): Whether to use biases. Default is ``True``.
        init_mean: The mean of the normal distribution for factor vectors
            initialization. Default is ``0``.
        init_std_dev: The standard deviation of the normal distribution for
            factor vectors initialization. Default is ``0.1``.
        lr_all: The learning rate for all parameters. Default is ``0.005``.
        reg_all: The regularization term for all parameters. Default is
            ``0.02``.
        lr_bu: The learning rate for :math:`b_u`. Takes precedence over
            ``lr_all`` if set. Default is ``None``.
        lr_bi: The learning rate for :math:`b_i`. Takes precedence over
            ``lr_all`` if set. Default is ``None``.
        lr_pu: The learning rate for :math:`p_u`. Takes precedence over
            ``lr_all`` if set. Default is ``None``.
        lr_qi: The learning rate for :math:`q_i`. Takes precedence over
            ``lr_all`` if set. Default is ``None``.
        reg_bu: The regularization term for :math:`b_u`. Takes precedence
            over ``reg_all`` if set. Default is ``None``.
        reg_bi: The regularization term for :math:`b_i`. Takes precedence
            over ``reg_all`` if set. Default is ``None``.
        reg_pu: The regularization term for :math:`p_u`. Takes precedence
            over ``reg_all`` if set. Default is ``None``.
        reg_qi: The regularization term for :math:`q_i`. Takes precedence
            over ``reg_all`` if set. Default is ``None``.
        random_state(int, RandomState instance from numpy, or ``None``):
            Determines the RNG(Random Number Generator) that will be used for initialization. If
            int, ``random_state`` will be used as a seed for a new RNG. This is
            useful to get the same initialization over multiple calls to
            ``fit()``.  If RandomState instance, this same instance is used as
            RNG. If ``None``, the current RNG from numpy is used.  Default is
            ``None``.
        verbose: If ``True``, prints the current epoch. Default is ``True``.
    Attributes:
        pu(numpy array of size (n_users, n_factors)): The user factors (only
            exists if ``fit()`` has been called)
        qi(numpy array of size (n_items, n_factors)): The item factors (only
            exists if ``fit()`` has been called)
        bu(numpy array of size (n_users)): The user biases (only
            exists if ``fit()`` has been called)
        bi(numpy array of size (n_items)): The item biases (only
            exists if ``fit()`` has been called)
    """

    def __init__(self, n_factors=100, n_epochs=20, biased=True, init_mean=0,
                 init_std_dev=.1, lr_all=.005,
                 reg_all=.02, lr_bu=None, lr_bi=None, lr_pu=None, lr_qi=None,
                 reg_bu=None, reg_bi=None, reg_pu=None, reg_qi=None,
                 random_state=None, verbose=True):

        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.biased = biased
        self.init_mean = init_mean
        self.init_std_dev = init_std_dev
        self.lr_bu = lr_bu if lr_bu is not None else lr_all
        self.lr_bi = lr_bi if lr_bi is not None else lr_all
        self.lr_pu = lr_pu if lr_pu is not None else lr_all
        self.lr_qi = lr_qi if lr_qi is not None else lr_all
        self.reg_bu = reg_bu if reg_bu is not None else reg_all
        self.reg_bi = reg_bi if reg_bi is not None else reg_all
        self.reg_pu = reg_pu if reg_pu is not None else reg_all
        self.reg_qi = reg_qi if reg_qi is not None else reg_all
        self.random_state = random_state
        self.verbose = verbose


    def fit(self, trainset):
        self.trainset = trainset
        self.sgd(trainset)

        return self

    def sgd(self, trainset):

        # The algorithm looks like this:
        #
        # for f in range(n_factors):
        #       for _ in range(n_iter):
        #           for u, i, r in all_ratings:
        #               err = r_ui - <p[u, :f+1], q[i, :f+1]>
        #               update p[u, f]
        #               update q[i, f]

        global_mean = self.trainset.global_mean

        lr_bu = self.lr_bu
        lr_bi = self.lr_bi
        lr_pu = self.lr_pu
        lr_qi = self.lr_qi

        reg_bu = self.reg_bu
        reg_bi = self.reg_bi
        reg_pu = self.reg_pu
        reg_qi = self.reg_qi

        # random number generator
        rng = np.random.RandomState(self.random_state) if self.random_state else np.random.RandomState()

        bu = np.zeros(trainset.n_users, np.double)
        bi = np.zeros(trainset.n_items, np.double)
        pu = rng.normal(self.init_mean, self.init_std_dev,
                        (trainset.n_users, self.n_factors))
        qi = rng.normal(self.init_mean, self.init_std_dev,
                        (trainset.n_items, self.n_factors))

        if not self.biased:
            global_mean = 0

        for current_epoch in range(self.n_epochs):
            if self.verbose:
                print("Processing epoch {}".format(current_epoch))
            for u, i, r in trainset.all_ratings():

                # compute current error
                dot = 0
                for f in range(self.n_factors):
                    dot += qi[i, f] * pu[u, f]
                err = r - (global_mean + bu[u] + bi[i] + dot)

                # update biases
                if self.biased:
                    bu[u] += lr_bu * (err - reg_bu * bu[u])
                    bi[i] += lr_bi * (err - reg_bi * bi[i])

                # update factors
                for f in range(self.n_factors):
                    puf = pu[u, f]
                    qif = qi[i, f]
                    pu[u, f] += lr_pu * (err * qif - reg_pu * puf)
                    qi[i, f] += lr_qi * (err * puf - reg_qi * qif)

        self.bu = bu
        self.bi = bi
        self.pu = pu
        self.qi = qi

    def estimate(self, u, i):

        is_user_rated = self.trainset.is_user_rated(u)
        is_item_rated = self.trainset.is_item_rated(i)

        if self.biased:
            est = self.trainset.global_mean

            if is_user_rated:
                est += self.bu[u]

            if is_item_rated:
                est += self.bi[i]

            if is_user_rated and is_item_rated:
                est += np.dot(self.qi[i], self.pu[u])

        else:
            if is_user_rated and is_item_rated:
                est = np.dot(self.qi[i], self.pu[u])
            else:
                raise InvalidPrediction('User and item are unknown.')

        return est

    def predict(self, uid, iid, r_ui=None, clip=True, verbose=True):
        """Compute the rating prediction for given user and item.

        The ``predict`` method converts raw ids to inner ids and then calls the
        ``estimate`` method. If the prediction is impossible (e.g. because the 
        user and/or the item is unkown), the prediction is set to global mean.

        Args:
            uid: (Raw) id of the user. See :ref:`this note<raw_inner_note>`.
            iid: (Raw) id of the item. See :ref:`this note<raw_inner_note>`.
            r_ui(float): The true rating :math:`r_{ui}`. Optional, default is
                ``None``.
            clip(bool): Whether to clip the estimation into the rating scale,
                that was set during dataset creation. For example, if
                :math:`\\hat{r}_{ui}` is :math:`5.5` while the rating scale is
                :math:`[1, 5]`, then :math:`\\hat{r}_{ui}` is set to :math:`5`.
                Same goes if :math:`\\hat{r}_{ui} < 1`.  Default is ``True``.
            verbose(bool): Whether to print details of the prediction.  Default
                is True.

        Returns:
            est : The estimated score

            - The (raw) user id ``uid``.
            - The (raw) item id ``iid``.
            - The true rating ``r_ui`` (:math:`\\hat{r}_{ui}`).
            - The estimated rating (:math:`\\hat{r}_{ui}`).
        """

        # Convert raw ids to inner ids
        try:
            iuid = self.trainset.to_inner_uid(uid)
        except ValueError:
            iuid = 'UKN__' + str(uid)
        try:
            iiid = self.trainset.to_inner_iid(iid)
        except ValueError:
            iiid = 'UKN__' + str(iid)

        try:
            est = self.estimate(iuid, iiid)
        except InvalidPrediction:
            est = self.trainset.global_mean

        # clip estimate into [lower_bound, higher_bound]
        if clip:
            lower_bound, higher_bound = self.trainset.rating_scale
            est = min(higher_bound, est)
            est = max(lower_bound, est)

        if verbose:
            print("uid:", uid, "| iid:", iid, "| actual rating:", r_ui, "| estimated rating:", '{est:1.2f}'.format(est=est))

        return est


