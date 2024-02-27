
from BaseGraphicalLasso import BaseGraphicalLasso
import penalty_functions as pf
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed


class SerialTVGL(BaseGraphicalLasso):
    def __init__(self, *args, **kwargs):
        super().__init__(processes=1, *args, **kwargs)

    def theta_update(self):
        for i in range(self.blocks):
            a = (self.z0s[i] + self.z1s[i] + self.z2s[i] - self.u0s[i] - self.u1s[i] - self.u2s[i]) / 3
            at = a.T
            m = self.eta * (a + at) / 2 - self.emp_cov_mat[i]
            d, q = np.linalg.eigh(m)
            sqrt_matrix = np.sqrt(d**2 + 4 / self.eta * np.ones(self.dimension))
            diagonal = np.diag(d) + np.diag(sqrt_matrix)
            self.thetas[i] = np.real(self.eta / 2 * q @ diagonal @ q.T)

    # def theta_update(self):
    #     def update_block(i):
    #         a = (self.z0s[i] + self.z1s[i] + self.z2s[i] - self.u0s[i] - self.u1s[i] - self.u2s[i]) / 3
    #         at = a.transpose()
    #         m = self.eta * (a + at) / 2 - self.emp_cov_mat[i]
    #         d, q = np.linalg.eig(m)
    #         qt = q.transpose()
    #         sqrt_matrix = np.sqrt(d**2 + 4 / self.eta * np.ones(self.dimension))
    #         diagonal = np.diag(d) + np.diag(sqrt_matrix)
    #         return np.real(self.eta / 2 * np.dot(np.dot(q, diagonal), qt))
    
    #     with ThreadPoolExecutor() as executor:
    #         futures = [executor.submit(update_block, i) for i in range(self.blocks)]
    #         for i, future in enumerate(as_completed(futures)):
    #             self.thetas[i] = future.result()

    def z_update(self):
        self.z0_update()
        self.z1_z2_update()

    # def z0_update(self):
    #     self.z0s = [pf.soft_threshold_odd(self.thetas[i] + self.u0s[i], self.lambd, self.rho) for i in range(self.blocks)]

    def z0_update(self):
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(pf.soft_threshold_odd, self.thetas[i] + self.u0s[i], self.lambd, self.rho) for i in range(self.blocks)]
            for i, future in enumerate(as_completed(futures)):
                self.z0s[i] = future.result()

    # def z1_z2_update(self):
    #     if self.penalty_function == "perturbed_node":
    #         for i in range(1, self.blocks):
    #             self.z1s[i-1], self.z2s[i] = pf.perturbed_node(self.thetas[i-1], self.thetas[i], self.u1s[i-1], self.u2s[i], self.beta, self.rho)
    #     else:
    #         aa = [self.thetas[i] - self.thetas[i-1] + self.u2s[i] - self.u1s[i-1] for i in range(1, self.blocks)]
    #         ee = [getattr(pf, self.penalty_function)(a, self.beta, self.rho) for a in aa]
    #         for i in range(1, self.blocks):
    #             summ = self.thetas[i-1] + self.thetas[i] + self.u1s[i-1] + self.u2s[i]
    #             self.z1s[i-1] = 0.5 * (summ - ee[i-1])
    #             self.z2s[i] = 0.5 * (summ + ee[i-1])

    def z1_z2_update(self):
        def update_for_perturbed_node(i):
            return pf.perturbed_node(self.thetas[i-1], self.thetas[i], self.u1s[i-1], self.u2s[i], self.beta, self.rho)
    
        def update_for_other_penalty_functions(i):
            a = self.thetas[i] - self.thetas[i-1] + self.u2s[i] - self.u1s[i-1]
            e = getattr(pf, self.penalty_function)(a, self.beta, self.rho)
            summ = self.thetas[i-1] + self.thetas[i] + self.u1s[i-1] + self.u2s[i]
            z1 = 0.5 * (summ - e)
            z2 = 0.5 * (summ + e)
            return z1, z2
    
        with ThreadPoolExecutor() as executor:
            futures = []
            if self.penalty_function == "perturbed_node":
                for i in range(1, self.blocks):
                    futures.append(executor.submit(update_for_perturbed_node, i))
            else:
                for i in range(1, self.blocks):
                    futures.append(executor.submit(update_for_other_penalty_functions, i))
    
            for future in as_completed(futures):
                result = future.result()
                if self.penalty_function == "perturbed_node":
                    i = futures.index(future) + 1  # Adjust index since range starts from 1
                    self.z1s[i-1], self.z2s[i] = result
                else:
                    i = futures.index(future) + 1  # Adjust index since range starts from 1
                    self.z1s[i-1], self.z2s[i] = result

    def u_update(self):
        for i in range(self.blocks):
            self.u0s[i] += self.thetas[i] - self.z0s[i]
        for i in range(1, self.blocks):
            self.u2s[i] += self.thetas[i] - self.z2s[i]
            self.u1s[i-1] += self.thetas[i-1] - self.z1s[i-1]
    
    # def u_update(self):
    #     def update_u0(i):
    #         return self.u0s[i] + self.thetas[i] - self.z0s[i]
    
    #     def update_u1_u2(i):
    #         u2 = self.u2s[i] + self.thetas[i] - self.z2s[i]
    #         u1 = self.u1s[i-1] + self.thetas[i-1] - self.z1s[i-1]
    #         return u1, u2
    
    #     with ThreadPoolExecutor() as executor:
    #         # Update u0s in parallel
    #         u0_futures = [executor.submit(update_u0, i) for i in range(self.blocks)]
    #         for i, future in enumerate(as_completed(u0_futures)):
    #             self.u0s[i] = future.result()
    
    #         # Update u1s and u2s in parallel, requires careful handling due to dependencies
    #         u1_u2_futures = [executor.submit(update_u1_u2, i) for i in range(1, self.blocks)]
    #         for i, future in enumerate(as_completed(u1_u2_futures)):
    #             self.u1s[i], self.u2s[i+1] = future.result()


