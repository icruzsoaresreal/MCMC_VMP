
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

class MCMC_VMP:

    def __init__(
            
        self, 
        
        Dados: np.ndarray, Categorias: int,

        nu_0: float, chi_0: np.ndarray

    ) -> None:

        # Dados observados
        
        self.X = Dados

        # Número de observações

        self.N = self.X.shape[0]

        # Número de categorias latentes

        self.K = Categorias

        # Parâmetros a priori

        self.nu_0 = nu_0

        self.chi_0 = chi_0

        # Dimensão dos parâmetros

        self.D = self.chi_0.shape[0]

        # Parâmetros a posteriori

        self.nu = np.zeros(shape = self.K)

        self.chi = np.zeros((self.K, self.D))

        # Parâmetros naturais

        self.eta = np.zeros((self.K, self.D))

        # Estatísticas suficientes

        self.r = np.zeros(shape = (self.N, self.K))

        self.N_barra = np.zeros(shape = self.K)

    def inicializa_r(self) -> None:

        self.r = np.random.dirichlet(alpha = self.K*[1/self.K], size = self.N)

    def atualiza_N_barra(self) -> None:

        self.N_barra = self.r.sum(axis = 0)

    def atualiza_chi(self) -> None:

        for k in range(self.K):

            self.chi[k] = np.vstack((self.X, self.X**2)) @ self.r[:, k]

            self.chi[k] += self.chi_0

    def q_Z(self, n: int, z) -> None:

        delta = 0

        for k in range(self.K):

            T_barra = np.array([self.X[n], self.X[n]**2])

            delta += np.dot(self.eta[k], T_barra*z[k])

        delta = np.exp(delta)

        return delta
    
    def atualiza_r(self) -> None:

        for n in range(self.N):

            Amostra = tfp.mcmc.sample_chain(

                num_results = 2, kernel = tfp.mcmc.RandomWalkMetropolis(

                    target_log_prob_fn = lambda z: self.q_Z(n = n, z = z)

                ), current_state = tf.zeros([self.K]), trace_fn = None,

            )

            self.r[n] = np.array(Amostra).mean(axis = 0)

    def q_eta(self, k:int, eta) -> None:

        q_eta = np.dot(eta, self.chi[k])

        q_eta = np.exp(q_eta + eta[0]**2/(4*eta[1]) + 1/2*np.log(-2*eta[1]))

        return q_eta
    
    def atualiza_eta(self) -> None:

        for k in range(self.K):

            Amostra = tfp.mcmc.sample_chain(

                num_results = 2, kernel = tfp.mcmc.RandomWalkMetropolis(

                    target_log_prob_fn = lambda eta: self.q_eta(k = k, eta = eta)

                ), current_state = np.array([np.float32(0), np.float32(-5)]), trace_fn = None,

            )

            self.eta[k] = np.array(Amostra).mean(axis = 0)

    def atualiza_modelo(self) -> None:

        self.atualiza_N_barra()

        self.atualiza_chi()

        self.atualiza_eta()

        self.atualiza_r()

    def estima_parametros(self) -> None:

        self.inicializa_r()

        self.atualiza_modelo()
        
        for i in range(10):

            self.atualiza_modelo()