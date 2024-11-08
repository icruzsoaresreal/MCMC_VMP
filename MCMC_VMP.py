
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

        # Parâmetros latentes

        self.r = np.zeros(shape = (self.N, self.K))

        self.N_barra = np.zeros(shape = self.K)

        # Parâmetros a priori

        self.nu_0 = nu_0

        self.chi_0 = chi_0

        # Dimensão dos parâmetros

        self.D = self.chi_0.shape[0]

        # Parâmetros a posteriori

        self.nu = np.zeros(shape = self.K)

        self.chi = np.zeros((self.D, self.K))

        # Parâmetros naturais

        self.eta = np.zeros((self.K, self.D))

        # Estatíscas suficientes

        self.u = np.zeros(shape = (self.D, self.K))

    def inicializa_modelo(self) -> None:

        alpha = self.K*[1/self.K]

        self.r = np.random.dirichlet(alpha = alpha, size = self.N)

    def atualiza_u(self) -> None:

        u_X = np.vstack((self.X, self.X**2))

        self.u = u_X @ self.r

    def atualiza_chi(self) -> None:

        self.chi = self.u

        self.chi += self.chi_0.reshape((self.D, 1))

    def atualiza_N_barra(self) -> None:

        self.N_barra = self.r.sum(axis = 0)

    def atualiza_nu(self) -> None:
    
        self.nu = self.nu_0 + self.N_barra

    def log_q_eta(self, k:int, eta: np.ndarray) -> None:

        log_q_eta = eta[0]**2/(4*eta[1])

        log_q_eta += 1/2*np.log(2*np.abs(eta[1]))

        log_q_eta *= self.nu[k]

        log_q_eta += np.dot(eta, self.chi[k])

        return log_q_eta
    
    def atualiza_eta(self) -> None:

        Ponto_inicial = np.array([np.float32(0), np.float32(-1/2)])

        for k in range(self.K):

            Amostra = tfp.mcmc.sample_chain(

                num_results = 10, current_state = Ponto_inicial,
                
                kernel = tfp.mcmc.RandomWalkMetropolis(

                    target_log_prob_fn = lambda eta: self.log_q_eta(k = k, eta = eta)

                ), trace_fn = None,

            )

            self.eta[k] = np.array(Amostra).mean(axis = 0)

    def q_Z(self, n: int, z) -> None:

        delta = 0

        for k in range(self.K):

            T_barra = np.array([self.X[n], self.X[n]**2])

            delta += np.dot(self.eta[k], T_barra*z[k])

        return delta
    
    def atualiza_r(self) -> None:

        for n in range(self.N):

            Amostra = tfp.mcmc.sample_chain(

                num_results = 5, kernel = tfp.mcmc.RandomWalkMetropolis(

                    target_log_prob_fn = lambda z: self.q_Z(n = n, z = z)

                ), current_state = tf.zeros([self.K]), trace_fn = None,

            )

            self.r[n] = np.array(Amostra).mean(axis = 0)

    def atualiza_modelo(self) -> None:

        self.atualiza_u()

        self.atualiza_chi()

        self.atualiza_N_barra()

        self.atualiza_nu()

        self.atualiza_eta()

        self.atualiza_r()

    def estima_parametros(self, max: int = 10) -> None:

        self.inicializa_modelo()

        self.atualiza_modelo()

        for i in range(max):

            self.atualiza_modelo()