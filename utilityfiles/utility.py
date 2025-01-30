import os 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from google.cloud import vision
from google.oauth2 import service_account
import requests
from IPython.display import display, Markdown

def get_image_description(image_path):
    """
    Use Google Vision API to get a description of the image.

    Parameters:
        image_path (str): The path to the image file.

    Returns:
        str: The description of the image.
    """
    json_key_path = 'sublime-index-************.json'
    credentials = service_account.Credentials.from_service_account_file(json_key_path)

    client = vision.ImageAnnotatorClient(credentials=credentials)
    
    with open(image_path, 'rb') as image_file:
        content = image_file.read()
    image = vision.Image(content=content)
    
    # Perform label detection on the image file
    response = client.label_detection(image=image)
    labels = response.label_annotations
    descriptions = [label.description for label in labels]

    if not descriptions:
        return "No description found"
    return ', '.join(descriptions)




def send_image_for_analysis(description):
    """
    Send the encoded image to ChatGPT for analysis.

    Parameters:
        encoded_image (str): The base64 encoded string of the image.

    Returns:
        dict: The JSON response from the API.
    """
    api_url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": "Bearer sk-*******************************************",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "gpt-3.5-turbo",
        "messages": [
            {"role": "system", "content": "Analyze the following image description:"},
            {"role": "user", "content": description}
        ]
        
    }
    response = requests.post(api_url, json=payload, headers=headers)
    return response.json()


class LinearRegression(object):
    
    def __init__(self, X, Y):
        self.X = np.array(X, dtype=np.float64)
        self.Y = np.array(Y, dtype=np.float64)
        
        if self.X.shape[0] != self.Y.shape[0]:
            raise ValueError("The arrays X and Y must have the same length.")

        self.n = len(self.X)
        self.m_X = np.mean(self.X)
        self.m_Y = np.mean(self.Y)
        self.SS_xx = np.sum(self.X**2) - self.n * self.m_X**2
        self.SS_xy = np.sum(self.Y * self.X) - self.n * self.m_Y * self.m_X
        
        if self.SS_xx == 0:
            raise ValueError("Division by zero error in coefficient calculation due to zero variance in X.")
        
        self.b_1 = self.SS_xy / self.SS_xx
        self.b_0 = self.m_Y - self.b_1 * self.m_X
        self.y_pred = self.b_0 + self.b_1 * self.X

        # Pre-compute MSE and R^2 score
        self._mse = np.mean((self.Y - self.y_pred)**2)
        self._r2_score = 1 - (np.sum((self.Y - self.y_pred)**2) / np.sum((self.Y - self.m_Y)**2))
        
    def plot_regression_line(self, title, xlabel='X', ylabel='Y', legend=False, legend_label=['Data1', 'Data2'], gpt_analysis=False):
        """
        Plot the regression line against the data points.

        Parameters:
            title (str): Title of the plot.
            legend (bool): Whether to display a legend.
            legend_label (list): Labels for the legend.
            gpt_analysis (bool): If true, save the plot for analysis and send to GPT.
        """
        plt.scatter(self.X, self.Y, color="m", marker="o", s=30)
        plt.plot(self.X, self.y_pred, color="g")
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        if legend:
            plt.legend(legend_label)
        
        plt.show()

        if gpt_analysis:
            directory = 'data/temp'
            os.makedirs(directory, exist_ok=True)
            filename = f"{np.random.randint(1000)}.png"
            filepath = os.path.join(directory, filename)

            plt.savefig(filepath)
            plt.close()  # Close the plot to free up memory

            # Encode the image
            description = get_image_description(filepath)
            
            temp_words = "This is a plot of a linear regression model, please analyze the plot based on the following description. \n"
            description = temp_words + description
            # Send the encoded image for analysis
            analysis_results = send_image_for_analysis(description)
            
            content = analysis_results['choices'][0]['message']['content']

            # print("Analysis Results:", content)
            display(Markdown(content))

            # Optional: Clean up the image file after sending
            os.remove(filepath)
                
    @property
    def coefficients(self):
        return self.b_0, self.b_1

    @property
    def mean_square_error(self):
        return np.mean((self.Y - self.y_pred)**2)

    @property
    def r2_score(self):
        return 1 - (np.sum((self.Y - self.y_pred)**2) / np.sum((self.Y - self.m_Y)**2))
    
    def predict(self, X):
        """
        Predict response values for given predictor values.

        Parameters:
        X (array-like): Predictor data points.

        Returns:
        array-like: Predicted response values.
        """
        return self.b_0 + self.b_1 * np.array(X)
    
    def rmse(self):
        """
        Calculate the root mean squared error of the model.

        Returns:
        float: The root mean squared error.
        """
        return np.sqrt(self.mean_square_error)
    
    def summary_table(self):
        """
        Generate a summary table of the linear regression model.

        Returns:
        pd.DataFrame: Summary table of the linear regression model.
        """
                
        # display the linear regression function
        print(f"Linear Regression Function: y = {self.b_0} + ({self.b_1}) * x")
        print(f"Mean Square Error (MSE): {self._mse:.4f}")
        print(f"R^2 Score: {self._r2_score:.4f}")
        print(f"Intercept (b0): {self.b_0:.4f}")
        print(f"Slope (b1): {self.b_1:.4f}")
        
        # plot the regression line
        self.plot_regression_line("Linear Regression", gpt_analysis=True)
                

class LogisticRegression(object):
    def __init__(self, X, Y, learning_rate=0.01, epochs=1000, verbose=False):
        self.X = np.array(X, dtype=np.float64).reshape(-1, 1)
        self.Y = np.array(Y, dtype=np.float64)
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = np.zeros(self.X.shape[1] + 1)  # +1 for the intercept
        self.verbose = verbose

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -250, 250)))  # Clip to avoid overflow

    def gradient_descent(self):
        X_b = np.hstack([np.ones((self.X.shape[0], 1)), self.X])  # add bias term
        for i in range(self.epochs):
            z = np.dot(X_b, self.weights)
            y_pred = self.sigmoid(z)
            errors = y_pred - self.Y
            gradients = np.dot(X_b.T, errors) / len(self.Y)
            self.weights -= self.learning_rate * gradients
            if i % (self.epochs // 10) == 0:  # Optionally log progress every 10% of the epochs
                cost = -np.mean(self.Y * np.log(np.maximum(y_pred, 1e-10)) + (1 - self.Y) * np.log(np.maximum(1 - y_pred, 1e-10)))
                print(f'Epoch {i}, Cost: {cost}')

    def predict_proba(self, X):
        X = np.array(X, dtype=np.float64).reshape(-1, 1)
        X_b = np.hstack([np.ones((X.shape[0], 1)), X])
        return self.sigmoid(np.dot(X_b, self.weights))

    def predict(self, X):
        return (self.predict_proba(X) > 0.5).astype(int)

    def get_coefficients(self):
        return self.weights[1:], self.weights[0]

    def evaluate(self, X, Y):
        predictions = self.predict(X)
        Y = np.array(Y, dtype=np.float64)
        accuracy = np.mean(predictions == Y)
        metrics = self.calculate_metrics(predictions, Y)
        metrics.update({'accuracy': accuracy})
        return metrics
    
    def calculate_metrics(self, predictions, actual):
        true_positive = np.sum((predictions == 1) & (actual == 1))
        true_negative = np.sum((predictions == 0) & (actual == 0))
        false_positive = np.sum((predictions == 1) & (actual == 0))
        false_negative = np.sum((predictions == 0) & (actual == 1))
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive > 0 else 0
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
        return {'precision': precision, 'recall': recall, 'f1_score': f1_score,
                'confusion_matrix': (true_positive, true_negative, false_positive, false_negative)}
        
    def plot_regression_line(self, title, xlabel, ylabel, legend=False, legend_label=['Data1', 'Data2'], gpt_analysis=False):
        """
        Plot the regression line against the data points.

        Parameters:
            title (str): Title of the plot.
            xlabel (str): Label for the x-axis.
            ylabel (str): Label for the y-axis.
            legend_labels (list): Labels for the legend.
            gpt_analysis (bool): If true, save the plot for analysis and send to GPT.
        """
        
        X_range = np.linspace(self.X.min(), self.X.max(), 300).reshape(-1, 1)
        probabilities = self.predict_proba(X_range)
        plt.scatter(self.X, self.Y, color="red", marker="o", s=30)
        plt.plot(X_range, probabilities, color="green")
        
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        if legend:
            plt.legend(legend_label)
        
        plt.show()
        
        if gpt_analysis:
            
            directory = 'data/temp'
            os.makedirs(directory, exist_ok=True)
            filename = f"{np.random.randint(1000,2000)}.png"
            filepath = os.path.join(directory, filename)

            plt.savefig(filepath)
            plt.close()  # Close the plot to free up memory

            # Encode the image
            description = get_image_description(filepath)
            
            temp_words = "This is a plot of a logistic regression model, please analyze the plot based on the following description. \n"
            description = temp_words + description
            # Send the encoded image for analysis
            analysis_results = send_image_for_analysis(description)
            
            content = analysis_results['choices'][0]['message']['content']

            # print("Analysis Results:", content)
            display(Markdown(content))

            # Optional: Clean up the image file after sending
            os.remove(filepath)
            
        
    
    def summary_table(self):
        """
        Generate a summary table of the logistic regression model.

        Returns:
        pd.DataFrame: Summary table of the logistic regression model.
        
        """
        coefficients, intercept = self.get_coefficients()
             
        # display the logistic regression function
        print(f"Logistic Regression Function: p(y = 1 | x) = σ({intercept:.4f} + {coefficients[0]:.4f} * x)")
        print(f"Learning Rate: {self.learning_rate}")
        print(f"Epochs: {self.epochs}")
        print(f"Intercept: {intercept:.4f}")
        print(f"Coefficient of X: {coefficients[0]:.4f}")
        
        # plot the regression line
        self.plot_regression_line("Logistic Regression", "Feature X", "Probability (Y)", gpt_analysis=True)


class BayesianLinearRegression(object):
    def __init__(self, alpha=1, beta=1):
        self.alpha = alpha
        self.beta = beta
        self.is_fitted = False

    def fit(self, X, Y):
        self.X = np.array(X)
        self.Y = np.array(Y)
        self.n = len(self.X)
        self.m_X, self.m_Y = np.mean(self.X), np.mean(self.Y)
        self.SS_xy = np.sum(self.Y * self.X) - self.n * self.m_Y * self.m_X
        self.SS_xx = np.sum(self.X * self.X) - self.n * self.m_X * self.m_X
        self.b_1 = self.SS_xy / self.SS_xx
        self.b_0 = self.m_Y - self.b_1 * self.m_X
        self.y_pred = self.b_0 + self.b_1 * self.X
        self.mean_square_error = np.sum((self.Y - self.y_pred)**2) / self.n
        self.r2_score = 1 - np.sum((self.Y - self.y_pred)**2) / np.sum((self.Y - self.m_Y)**2)
        self.posterior_distribution()
        self.is_fitted = True
        return self

    def posterior_distribution(self, xlabel, ylabel, title):
        self.alpha_n = self.alpha + self.n / 2
        self.beta_n = self.beta + 0.5 * (np.sum(self.Y**2) - 2 * self.b_1 * np.sum(self.X * self.Y) + self.b_1**2 * np.sum(self.X**2))
        self.mean = self.beta_n / (self.alpha_n - 1)
        self.variance = self.beta_n / ((self.alpha_n - 1) * (self.alpha_n - 2))
        self.predictive_mean, self.predictive_variance = self.get_predictive_distribution(self.X)

    def plot_regression_line(self, title, legend = False, legend_label = ['xxxxxx', 'xxxxxxx']):
        plt.scatter(self.X, self.Y, color = "m", marker = "o", s = 30)
        plt.plot(self.X, self.b_0 + self.b_1*self.X, color = "g")
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        if legend:
            plt.legend(legend_label)
        plt.show()
    
    def fit(self, X, Y):
        """

        Fit the Bayesian linear regression model.
        
        Parameters:
        X (array-like): Independent variable(s).
        Y (array-like): Dependent variable.

        Returns:
        self: Returns an instance of self.
        
        """
        
        if len(X) == 0 or len(Y) == 0:
            raise ValueError("Input arrays cannot be empty.")
        if len(X) != len(Y):
            raise ValueError("Input arrays must have the same length.")
        
        self.X = np.array(X)
        self.Y = np.array(Y)
        self.n = len(self.X)
        self.m_X, self.m_Y = np.mean(self.X), np.mean(self.Y)
        self.SS_xy = np.sum(self.Y * self.X) - self.n * self.m_Y * self.m_X
        self.SS_xx = np.sum(self.X * self.X) - self.n * self.m_X * self.m_X
        self.b_1 = self.SS_xy / self.SS_xx
        self.b_0 = self.m_Y - self.b_1 * self.m_X
        self.y_pred = self.b_0 + self.b_1 * self.X
        self.mean_square_error = np.sum((self.Y - self.y_pred)**2) / self.n
        self.r2_score = 1 - np.sum((self.Y - self.y_pred)**2) / np.sum((self.Y - self.m_Y)**2)
        self.posterior_distribution()
        self.is_fitted = True
        return self

    def get_predictive_distribution(self, X):
        if not self.is_fitted:
            raise Exception("The model must be fitted before predicting.")
        mean = self.b_0 + self.b_1 * X
        variance = self.beta_n / (self.alpha_n - 1) + self.variance
        return mean, variance

    def get_predictive_interval(self, X, alpha = 0.05):
        mean, variance = self.get_predictive_distribution(X)
        lower_bound = mean - np.sqrt(variance)*1.96
        upper_bound = mean + np.sqrt(variance)*1.96
        return lower_bound, upper_bound
    
    def plot_predictive_interval(self, xlabel, ylabel, title, alpha = 0.05):
        lower_bound, upper_bound = self.get_predictive_interval(self.X, alpha)
        plt.scatter(self.X, self.Y, color = "m", marker = "o", s = 30)
        plt.plot(self.X, self.b_0 + self.b_1*self.X, color = "g")
        plt.fill_between(self.X, lower_bound, upper_bound, color = 'g', alpha = 0.1)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.show()
        
    def plot_predictive_distribution(self, xlabel, ylabel, title):
        mean, variance = self.get_predictive_distribution(self.X)
        lower_bound = mean - np.sqrt(variance)*1.96
        upper_bound = mean + np.sqrt(variance)*1.96
        plt.scatter(self.X, self.Y, color = "m", marker = "o", s = 30)
        plt.plot(self.X, self.b_0 + self.b_1*self.X, color = "g")
        plt.fill_between(self.X, lower_bound, upper_bound, color = 'g', alpha = 0.1)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.show()
    
    def plot_predictive_interval(self, xlabel, ylabel, title, alpha = 0.05):
        lower_bound, upper_bound = self.get_predictive_interval(self.X, alpha)
        plt.scatter(self.X, self.Y, color = "m", marker = "o", s = 30)
        plt.plot(self.X, self.b_0 + self.b_1*self.X, color = "g")
        plt.fill_between(self.X, lower_bound, upper_bound, color = 'g', alpha = 0.1)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.show()
        

class MCMC_Simulate(object):
    """
    A class for performing Markov Chain Monte Carlo (MCMC) simulations.
    
    Attributes:
        target_distribution (callable): Probability density function of the target distribution.
        proposal_distribution (callable): Proposal distribution function to suggest new samples.
        initial_value (float): Starting point for the MCMC simulation.
    """
    def __init__(self, target_distribution, proposal_distribution, initial_value, n_samples=1000000):
        self.target_distribution = target_distribution
        self.proposal_distribution = proposal_distribution
        self.current_value = initial_value
        self.n_samples = n_samples
        self.samples = []
        self.acceptance_rate = 0
        self.generate_samples()
        
    def generate_samples(self):
        """
        Generate samples using the MCMC method with the Metropolis-Hastings algorithm.
        """
        n_accepted = 0
        current_distribution_value = self.target_distribution(self.current_value)
        
        for _ in range(self.n_samples):
            proposed_value = self.proposal_distribution(self.current_value)
            proposed_distribution_value = self.target_distribution(proposed_value)
            acceptance_ratio = proposed_distribution_value / current_distribution_value
            
            if acceptance_ratio > np.random.uniform():
                self.current_value = proposed_value
                current_distribution_value = proposed_distribution_value
                n_accepted += 1
            self.samples.append(self.current_value)
            
        self.acceptance_rate = n_accepted / self.n_samples

    def plot_samples(self, xlabel, ylabel, title="MCMC Sampling Trace"):
        """
        Plot the trace of sampled values.
        """
        plt.plot(self.samples)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.show()
        
    def get_acceptance_rate(self):
        """
        Return the acceptance rate of the sampling process.
        """
        return self.acceptance_rate
    
    def get_samples(self):
        """
        Return the list of sampled values.
        """
        return self.samples
    
    def get_mean(self):
        """
        Return the mean of the sampled values.
        """
        return np.mean(self.samples)
    
    def get_variance(self):
        """
        Return the variance of the sampled values.
        """
        return np.var(self.samples)
    
    def get_quantile(self, q):
        """
        Return the q-th quantile of the sampled values.
        """
        return np.percentile(self.samples, q * 100)
    
    def get_hpd(self, alpha=0.05):
        """
        Calculate and return the Highest Posterior Density (HPD) interval for the given alpha.
        """
        lower_bound = self.get_quantile(alpha / 2)
        upper_bound = self.get_quantile(1 - alpha / 2)
        return lower_bound, upper_bound
    
    def plot_hpd(self, xlabel, ylabel, title="HPD Interval", alpha=0.05):
        """
        Plot the trace of sampled values along with the HPD interval.
        """
        lower_bound, upper_bound = self.get_hpd(alpha)
        plt.plot(self.samples)
        plt.axhline(lower_bound, color='r', linestyle='--', label='HPD Lower')
        plt.axhline(upper_bound, color='r', linestyle='--', label='HPD Upper')
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend()
        plt.show()
    
    def get_autocorrelation(self, xlabel, ylabel, lag=1):
        """
        Calculate and return the autocorrelation of sampled values at a specified lag.
        """
        return np.corrcoef(self.samples[:-lag], self.samples[lag:])[0, 1]
    
    def get_autocorrelation_plot(self, title="Autocorrelation Plot"):
        """
        Plot the autocorrelation for different lags up to half the number of samples.
        """
        autocorrelation = [self.get_autocorrelation(i) for i in range(1, len(self.samples) // 2)]
        plt.plot(autocorrelation)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.show()

    def get_gelman_rubin(self, n_chains=4):
        """
        Calculate and return the Gelman-Rubin diagnostic using multiple chains.
        """
        chains = [MCMC_Simulate(self.target_distribution, self.proposal_distribution, np.random.uniform()) for _ in range(n_chains)]
        n_samples = len(chains[0].get_samples())
        means = np.array([chain.get_mean() for chain in chains])
        variances = np.array([chain.get_variance() for chain in chains])
        mean_of_means = np.mean(means)
        B = n_samples / (n_chains - 1) * np.sum((means - mean_of_means)**2)
        W = np.mean(variances)
        var_plus = (n_samples - 1) / n_samples * W + B / n_samples
        R = np.sqrt(var_plus / W)
        return R
