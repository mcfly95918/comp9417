import numpy as np


def rmse(predictions, verbose=True):
    """Compute RMSE (Root Mean Squared Error).
    .. math::
        \text{RMSE} = \sqrt{\frac{1}{|\hat{R}|} \sum_{\hat{r}_{ui} \in
        \hat{R}}(r_{ui} - \hat{r}_{ui})^2}.
    Args:
        predictions (list of tuples, of which the first element is actual 
        rating, the second is predicted rating).
        verbose: If True, will print computed value. Default is ``True``.
    Returns:
        The Root Mean Squared Error of predictions.
    Raises:
        ValueError: When ``predictions`` is empty.
    """

    if not predictions:
        raise ValueError('Prediction list is empty.')

    mse = np.mean([float((true_r - est)**2)
                   for (true_r, est) in predictions])
    rmse_ = np.sqrt(mse)

    if verbose:
        print('RMSE: {0:1.4f}'.format(rmse_))

    return rmse_


def mae(predictions, verbose=True):
    """Compute MAE (Mean Absolute Error).
    .. math::
        \text{MAE} = \frac{1}{|\hat{R}|} \sum_{\hat{r}_{ui} \in
        \hat{R}}|r_{ui} - \hat{r}_{ui}|
    Args:
        predictions (list of tuples, of which the first element is actual 
        rating, the second is predicted rating).
        verbose: If True, will print computed value. Default is ``True``.
    Returns:
        The Mean Absolute Error of predictions.
    Raises:
        ValueError: When ``predictions`` is empty.
    """

    if not predictions:
        raise ValueError('Prediction list is empty.')

    mae_ = np.mean([float(abs(true_r - est))
                    for (true_r, est) in predictions])

    if verbose:
        print('MAE:  {0:1.4f}'.format(mae_))

    return mae_