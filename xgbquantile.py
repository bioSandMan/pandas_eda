
import numpy as np


class XGBQuantile():
    """
    # Example usage
    alpha = 0.95 #@param {type:"number"}

    X_train,y_train,X_test,y_test = generate_data()


    regressor = GradientBoostingRegressor(n_estimators=250, max_depth=3,
                                    learning_rate=.1, min_samples_leaf=9,
                                    min_samples_split=9)
    y_pred = regressor.fit(X_train,y_train).predict(X_test)
    regressor.set_params(loss='quantile', alpha=1.-alpha)
    y_lower = collect_prediction(X_train,y_train,X_test,
              y_test,estimator=regressor,alpha=1.-alpha,
              model_name="Gradient Boosting")
    regressor.set_params(loss='quantile', alpha=alpha)
    y_upper = collect_prediction(X_train,y_train,X_test,
              y_test,estimator=regressor,alpha=alpha,
              model_name="Gradient Boosting")
    fig = plt.figure(figsize=(12,6))

    plt.subplot(211)
    plt.title("Prediction Interval Gradient Boosting")
    plot_result(X_train,y_train,X_test,y_test,y_upper,y_lower)


    regressor = XGBRegressor(n_estimators=250,max_depth=3,reg_alpha=5,
                             reg_lambda=1,gamma=0.5)
    y_pred = regressor.fit(X_train,y_train).predict(X_test)

    regressor = XGBQuantile(n_estimators=100,max_depth = 3,
                            reg_alpha =5.0,gamma = 0.5,reg_lambda =1.0 )
    regressor.set_params(quant_alpha=1.-alpha,quant_delta=1.0,quant_thres=5.0,quant_var=3.2)

    y_lower = collect_prediction(X_train,y_train,X_test,
              y_test,estimator=regressor,alpha=1.-alpha,
              model_name="Quantile XGB")
    regressor.set_params(quant_alpha=alpha,quant_delta=1.0,
                         quant_thres=6.0,quant_var = 4.2)
    y_upper = collect_prediction(X_train,y_train,X_test,
                                 y_test,estimator=regressor,
                                 alpha=alpha,model_name="Quantile XGB")

    plt.subplot(212)
    plt.title("Prediction Interval XGBoost")
    plot_result(X_train,y_train,X_test,y_test,y_upper,y_lower)

    """
    def __init__(self,
                 estimator=None,
                 quant_alpha=0.95,
                 quant_delta=1.0,
                 quant_thres=1.0,
                 quant_var=1.0,):

        self.estimator = estimator
        self.quant_alpha = quant_alpha
        self.quant_delta = quant_delta
        self.quant_thres = quant_thres
        self.quant_var = quant_var

    def score(self, X, y):
        y_pred = self.estimator.predict(X)
        score = XGBQuantile.quantile_score(y, y_pred, self.quant_alpha)
        score = 1./score
        return score

    @staticmethod
    def quantile_loss(y_true, y_pred, alpha, delta, threshold, var):
        x = y_true - y_pred
        grad = (x < (alpha-1.0)*delta)*(1.0-alpha) - ((x >= (alpha-1.0)
                                                       * delta) & (x < alpha*delta))*x/
                                                       delta-alpha*(x > alpha*delta)
        hess = ((x >= (alpha-1.0)*delta) & (x < alpha*delta))/delta

        grad = (np.abs(x) < threshold)*grad - (np.abs(x) >= threshold) * \
               (2*np.random.randint(2, size=len(y_true)) - 1.0)*var
        hess = (np.abs(x) < threshold)*hess + (np.abs(x) >= threshold)
        return grad, hess
  
    @staticmethod
    def original_quantile_loss(y_true,y_pred,alpha,delta):
        x = y_true - y_pred
        grad = (x<(alpha-1.0)*delta)*(1.0-alpha)-((x>=(alpha-1.0)*delta)& (x<alpha*delta) )*x/delta-alpha*(x>alpha*delta)
        hess = ((x>=(alpha-1.0)*delta)& (x<alpha*delta) )/delta 
        return grad,hess

    @staticmethod
    def quantile_score(y_true, y_pred, alpha):
        score = XGBQuantile.quantile_cost(x=y_true-y_pred,alpha=alpha)
        score = np.sum(score)
        return score
    
    @staticmethod
    def quantile_cost(x, alpha):
        return (alpha-1.0)*x*(x<0)+alpha*x*(x>=0)
  
    @staticmethod
    def get_split_gain(gradient,hessian,l=1):
        split_gain = list()
        for i in range(gradient.shape[0]):
            split_gain.append(np.sum(gradient[:i])**2/(np.sum(hessian[:i])+l)+np.sum(gradient[i:])**2/(np.sum(hessian[i:])+l)-np.sum(gradient)**2/(np.sum(hessian)+l) )

        return np.array(split_gain)

    def collect_prediction(self, X_test, y_test, estimator, alpha, model_name):
        y_pred = estimator.predict(X_test)
        print( "{model_name} alpha = {alpha:.2f},score = {score:.1f}".format(model_name=model_name, alpha=alpha , score= XGBQuantile.quantile_score(y_test, y_pred, alpha)) )
        return y_pred

    def plot_result(self, X_train, y_train, X_test, y_test, y_upper, y_lower):
        y_pred = self.collect_prediction(X_test)
        plt.plot(X_test,y_test, 'g:', label=u'$f(x) = x\,\sin(x)$')
        plt.plot(X_train,y_train, 'b.', markersize=10, label=u'Observations')
        plt.plot(X_test, y_pred, 'r-', label=u'Prediction')
        plt.plot(X_test, y_upper, 'k-')
        plt.plot(X_test, y_lower, 'k-')
        plt.fill(np.concatenate([X_test, X_test[::-1]]),
            np.concatenate([y_upper, y_lower[::-1]]),
            alpha=.5, fc='b', ec='None', label='90% prediction interval')
        plt.xlabel('$x$')
        plt.ylabel('$f(x)$')
        plt.ylim(-10 , 20)
        plt.legend(loc='upper left')
