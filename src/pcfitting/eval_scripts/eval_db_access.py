import sqlite3
from typing import Optional

#fit_or_eval -> True = fit, False = eval

class EvalDbAccess:

    def __init__(self, dbpath):
        self._con = sqlite3.connect(dbpath)

    def insert_options_em(self,
                          n_sample_points: int = -1,
                          termination_criterion: str = "MaxIter(100)",
                          init_method: str = "randnormpos",
                          dtype: str = "float32",
                          eps: float = 1e-7,
                          is_eps_relative: bool = True) -> int:
        sql = "INSERT INTO OptionsEM(n_sample_points, termination_criterion, init_method, dtype, eps, is_eps_relative) VALUES (?,?,?,?,?,?)"
        cur = self._con.cursor()
        cur.execute(sql, (n_sample_points, termination_criterion, init_method, dtype, eps, is_eps_relative))
        self._con.commit()
        return cur.lastrowid

    def insert_options_eckart_hp(self,
                                 n_gaussians_per_node: int,
                                 n_levels: int,
                                 termination_criterion: str = "MaxIter(20)",
                                 init_method: str = "bb",
                                 dtype: str = "float32",
                                 eps: float = 1e-7,
                                 is_eps_relative: bool = True) -> int:
        sql = "INSERT INTO OptionsEckartHP(n_gaussians_per_node, n_levels, termination_criterion, init_method, dtype, eps, is_eps_relative) VALUES (?,?,?,?,?,?,?)"
        cur = self._con.cursor()
        cur.execute(sql, (n_gaussians_per_node, n_levels, termination_criterion, init_method, dtype, eps, is_eps_relative))
        self._con.commit()
        return cur.lastrowid

    def insert_options_eckart_sp(self,
                                 n_gaussians_per_node: int,
                                 n_levels: int,
                                 partition_threshold: float,
                                 termination_criterion: str = "MaxIter(20)",
                                 init_method: str = "randnormpos",
                                 dtype: str = "float32",
                                 eps: float = "1e-7",
                                 is_eps_relative: bool = True) -> int:
        sql = "INSERT INTO OptionsEckartSP(n_gaussians_per_node, n_levels, partition_threshold, termination_criterion, init_method, dtype, eps, is_eps_relative) VALUES (?,?,?,?,?,?,?,?)"
        cur = self._con.cursor()
        cur.execute(sql,
                    (n_gaussians_per_node, n_levels, partition_threshold, termination_criterion, init_method, dtype, eps, is_eps_relative))
        self._con.commit()
        return cur.lastrowid

    def insert_options_preiner(self,
                               alpha: float = 2.0,
                               pointpos: bool = True,
                               stdev: float = 0.01,
                               iso: bool = False,
                               inittype: str = "fixed",
                               knn: int = 8,
                               fixeddist: float = 0.1,
                               weighted: bool = False,
                               levels: int = 20,
                               reductionfactor: float = 3,
                               fixedngaussians: int = 0,
                               avoidorphans: bool = False) -> int:
        sql = "INSERT INTO OptionsPreiner(alpha, pointpos, stdev, iso, inittype, knn, fixeddist, weighted, levels, " \
              "reductionfactor, fixedngaussians, avoidorphans) VALUES (?,?,?,?,?,?,?,?,?,?,?,?) "
        cur = self._con.cursor()
        cur.execute(sql, (alpha, pointpos, stdev, iso, inittype, knn, fixeddist, weighted, levels, reductionfactor,
                          fixedngaussians, avoidorphans))
        self._con.commit()
        return cur.lastrowid

    def insert_density_eval(self,
                            avg_log: float,
                            avg_log_scaled: float,
                            avg: float,
                            avg_scaled: float,
                            stdev: float,
                            stdev_scaled: float,
                            cv: float,
                            ev_correction: Optional[float],
                            run_id: int,
                            fit_or_eval: bool,
                            pointcount: int) -> int:
        sql = "INSERT INTO DensityEval(avg_log, avg_log_scaled, avg, avg_scaled, stdev, stdev_scaled, cv, " \
              "ev_correction, run, fit_or_eval, pointcount) VALUES (?,?,?,?,?,?,?,?,?,?, ?) "
        cur = self._con.cursor()
        cur.execute(sql, (avg_log, avg_log_scaled, avg, avg_scaled, stdev, stdev_scaled, cv, ev_correction, run_id,
                          fit_or_eval, pointcount))
        self._con.commit()
        return cur.lastrowid

    def insert_distance_eval(self,
                             stg_rmsd: float,
                             stg_rmsd_scaled: float,
                             stg_md: float,
                             stg_md_scaled: float,
                             stg_std: float,
                             stg_std_scaled: float,
                             stg_cv: float,
                             stg_maxd: float,
                             stg_maxd_scaled: float,
                             gts_rmsd: float,
                             gts_rmsd_scaled: float,
                             gts_md: float,
                             gts_md_scaled: float,
                             gts_std: float,
                             gts_std_scaled: float,
                             gts_cv: float,
                             gts_maxd: float,
                             gts_maxd_scaled: float,
                             chamfer: float,
                             chamfer_scaled: float,
                             hausdorff: float,
                             hausdorff_scaled: float,
                             run: int,
                             fit_or_eval: bool,
                             pointcount: int) -> int:
        sql = "INSERT INTO DistanceEval(stg_rmsd, stg_rmsd_scaled, stg_md, stg_md_scaled, stg_std, " \
              "stg_std_scaled, stg_cv, stg_maxd, stg_maxd_scaled,gts_rmsd,gts_rmsd_scaled,gts_md," \
              "gts_md_scaled,gts_std,gts_std_scaled,gts_cv,gts_maxd,gts_maxd_scaled,chamfer," \
              "chamfer_scaled,hausdorff,hausdorff_scaled,run,fit_or_eval, pointcount) VALUES (?,?,?,?,?,?,?,?,?,?,?,?," \
              "?,?,?,?,?,?,?,?,?,?,?,?,?) "
        cur = self._con.cursor()
        cur.execute(sql, (stg_rmsd, stg_rmsd_scaled, stg_md, stg_md_scaled, stg_std, stg_std_scaled, stg_cv, stg_maxd,
                          stg_maxd_scaled,gts_rmsd,gts_rmsd_scaled,gts_md,gts_md_scaled,gts_std,gts_std_scaled,gts_cv,
                          gts_maxd,gts_maxd_scaled,chamfer,chamfer_scaled,hausdorff,hausdorff_scaled,run,fit_or_eval,
                          pointcount))
        self._con.commit()
        return cur.lastrowid

    def insert_stat_eval(self,
                         avg_trace: float,
                         std_traces: float,
                         avg_l_ev: float,
                         avg_m_ev: float,
                         avg_s_ev: float,
                         std_l_ev: float,
                         std_m_ev: float,
                         std_s_ev: float,
                         min_ev: float,
                         avg_amp: float,
                         std_amp: float,
                         avg_det: float,
                         std_det: float,
                         avg_weight: float,
                         std_weight: float,
                         sum_of_weights: float,
                         n_zero_gaussians: int,
                         n_invalid_gaussians: int,
                         run: int) -> int:
        sql = "INSERT INTO StatEval(avg_trace,std_traces,avg_l_ev,avg_m_ev,avg_s_ev,std_l_ev,std_m_ev,std_s_ev," \
              "min_ev,avg_amp,std_amp,avg_det,std_det,avg_weight,std_weight,sum_of_weights,n_zero_gaussians," \
              "n_invalid_gaussians,run) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)"
        cur = self._con.cursor()
        cur.execute(sql, (avg_trace,std_traces,avg_l_ev,avg_m_ev,avg_s_ev,std_l_ev,std_m_ev,std_s_ev,min_ev,avg_amp,
                          std_amp,avg_det,std_det,avg_weight,std_weight,sum_of_weights,n_zero_gaussians,
                          n_invalid_gaussians,run))
        self._con.commit()
        return cur.lastrowid

    def insert_run(self,
                   modelfile: str,
                   nr_fit_points: int,
                   n_gaussians_should: int,
                   n_gaussians_is: int,
                   method: str,
                   method_options: int,
                   execution_time: float) -> int:
        sql = "INSERT INTO Run(modelfile, nr_fit_points, n_gaussians_should, n_gaussians_is, method, " \
              "method_options, execution_time) VALUES (?, ?, ?, ?, ?, ?, ?)"
        cur = self._con.cursor()
        cur.execute(sql, (modelfile, nr_fit_points, n_gaussians_should, n_gaussians_is, method,
                          method_options, execution_time))
        self._con.commit()
        return cur.lastrowid

    def has_em_run(self,
                   modelfile: str,
                   nr_fit_points: int,
                   n_gaussians_should: int,
                   n_sample_points: int,
                   termination_criterion: str,
                   init_method: str,
                   dtype: str,
                   eps: float,
                   is_eps_relative: bool) -> bool:
        sql = "SELECT Run.ID FROM Run JOIN OptionsEM ON Run.method = 'EM' AND Run.method_options = OptionsEM.ID WHERE " \
              "Run.modelfile = ? AND Run.nr_fit_points = ? AND Run.n_gaussians_should = ? " \
              "AND OptionsEM.n_sample_points = ? AND OptionsEM.termination_criterion = ? AND OptionsEM.init_method = ? " \
              "AND OptionsEM.dtype = ? AND OptionsEM.eps = ? AND OptionsEM.is_eps_relative = ?"
        cur = self._con.cursor()
        cur.execute(sql, (modelfile, nr_fit_points, n_gaussians_should, n_sample_points,
                          termination_criterion, init_method, dtype, eps, is_eps_relative))
        list = cur.fetchall()
        return len(list) > 0

    def has_eck_hp_run(self,
                       modelfile: str,
                       nr_fit_points: int,
                       n_gaussians_should: int,
                       n_gaussians_per_node: int,
                       n_levels: int,
                       termination_criterion: str,
                       init_method: str,
                       dtype: str,
                       eps: float,
                       is_eps_relative: bool) -> bool:
        sql = "SELECT Run.ID FROM Run JOIN OptionsEckartHP ON Run.method = 'EckartHP' AND Run.method_options = " \
              "OptionsEckartHP.ID WHERE Run.modelfile = ? AND Run.nr_fit_points = ? AND" \
              " Run.n_gaussians_should = ? AND OptionsEckartHP.n_gaussians_per_node = ? AND OptionsEckartHP.n_levels " \
              "= ? AND OptionsEckartHP.termination_criterion = ? AND OptionsEckartHP.init_method = ? " \
              "AND OptionsEckartHP.dtype = ? AND OptionsEckartHP.eps = ? AND OptionsEckartHP.is_eps_relative = ?"
        cur = self._con.cursor()
        cur.execute(sql, (modelfile, nr_fit_points, n_gaussians_should, n_gaussians_per_node, n_levels,
                          termination_criterion, init_method, dtype, eps, is_eps_relative))
        list = cur.fetchall()
        return len(list) > 0

    def has_eck_sp_run(self,
                       modelfile: str,
                       nr_fit_points: int,
                       n_gaussians_should: int,
                       n_gaussians_per_node: int,
                       n_levels: int,
                       partition_threshold: float,
                       termination_criterion: str,
                       init_method: str,
                       dtype: str,
                       eps: float,
                       is_eps_relative: bool) -> bool:
        sql = "SELECT Run.ID FROM Run JOIN OptionsEckartSP ON Run.method = 'EckartSP' AND Run.method_options = " \
              "OptionsEckartSP.ID WHERE Run.modelfile = ? AND Run.nr_fit_points = ? AND" \
              " Run.n_gaussians_should = ? AND OptionsEckartSP.n_gaussians_per_node = ? AND OptionsEckartSP.n_levels " \
              "= ? AND OptionsEckartSP.partition_threshold = ? AND OptionsEckartSP.termination_criterion = ? AND " \
              "OptionsEckartSP.init_method = ? AND OptionsEckartSP.dtype = ? AND OptionsEckartSP.eps = ? AND " \
              "OptionsEckartSP.is_eps_relative = ?"
        cur = self._con.cursor()
        cur.execute(sql, (modelfile, nr_fit_points, n_gaussians_should, n_gaussians_per_node, n_levels,
                          partition_threshold, termination_criterion, init_method, dtype, eps, is_eps_relative))
        list = cur.fetchall()
        return len(list) > 0

    def has_preiner_run(self,
                       modelfile: str,
                       nr_fit_points: int,
                       n_gaussians_should: int,
                       alpha: float = 2.0,
                       pointpos: bool = True,
                       stdev: float = 0.01,
                       iso: bool = False,
                       inittype: str = "fixed",
                       knn: int = 8,
                       fixeddist: float = 0.1,
                       weighted: bool = False,
                       levels: int = 20,
                       reductionfactor: float = 3,
                       fixedngaussians: int = 0,
                       avoidorphans: bool = False) -> bool:
        sql = "SELECT Run.ID FROM Run JOIN OptionsPreiner ON Run.method = 'Preiner' AND Run.method_options = " \
              "OptionsPreiner.ID WHERE Run.modelfile = ? AND Run.nr_fit_points = ? AND" \
              " Run.n_gaussians_should = ? AND OptionsPreiner.alpha = ? AND OptionsPreiner.pointpos = ? AND " \
              "OptionsPreiner.stdev = ? AND OptionsPreiner.iso = ? AND OptionsPreiner.inittype = ? AND " \
              "OptionsPreiner.knn = ? AND OptionsPreiner.fixeddist = ? AND OptionsPreiner.weighted = ? AND " \
              "OptionsPreiner.levels = ? AND OptionsPreiner.reductionfactor = ? AND OptionsPreiner.fixedngaussians " \
              "= ? AND OptionsPreiner.avoidorphans = ?"
        cur = self._con.cursor()
        cur.execute(sql, (modelfile, nr_fit_points, n_gaussians_should, alpha, pointpos, stdev,
                          iso, inittype, knn, fixeddist, weighted, levels, reductionfactor, fixedngaussians,
                          avoidorphans))
        list = cur.fetchall()
        return len(list) > 0

    def __del__(self):
        self._con.close()