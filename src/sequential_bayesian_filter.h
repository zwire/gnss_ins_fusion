#ifndef SEQUENTIAL_BAYESIAN_FILTER_H_
#define SEQUENTIAL_BAYESIAN_FILTER_H_

#include <vector>
#include <float.h>
#include <random>
#include <functional>
#include <Eigen/Core>
#include <Eigen/LU>
#include <Eigen/Cholesky>

using namespace std;
using namespace Eigen;

////////////////////////////////////////////////////////////////

struct TransitionVariables final
{
  VectorXd X;
  VectorXd U;
  double dt;
  explicit TransitionVariables(const VectorXd& x, const VectorXd& u, const double& dt)
  {
    this->X = x;
    this->U = u;
    this->dt = dt;
  }
};

struct ObservationVariables final
{
  VectorXd X;
  explicit ObservationVariables(const VectorXd& x)
  {
    this->X = x;
  }
};

// interface
class SequentialBayesianFilter
{
  // -- linear state expression --
  // Xt+1 = (AXt + BUt) * dt + W
  // Yt+1 = CXt+1 + V
  // -- non-linear state expression --
  // Xt+1 = nonlinear_transition_function(Xt, Ut, W, dt)
  // Yt+1 = nonlinear_observation_function(Xt+1, V)
  // where: W ~ Q, V ~ R

protected:
  SequentialBayesianFilter(){}
  int _k;       // state vector size
  int _m;       // observation vector size
  int _n;       // control vector size

public:
  double dt;    // time span
  MatrixXd A;   // the state-transition model
  MatrixXd B;   // the control-input model
  MatrixXd C;   // the observation model
  MatrixXd Q;   // the covariance of the process noise
  MatrixXd R;   // the covariance of the observation noise
  MatrixXd P;   // the a posteriori estimate covariance matrix
  VectorXd X;   // the a posteriori state estimate
  function<VectorXd (TransitionVariables)> transition_func;
  function<VectorXd (ObservationVariables)> observation_func;

  explicit SequentialBayesianFilter(
    const VectorXd& x0,
    const int& observation_vector_size,
    const int& control_vector_size
  )
  {
    _k = x0.size();
    _m = observation_vector_size;
    _n = control_vector_size;
    X = VectorXd(x0);
    A = MatrixXd::Identity(_k, _k);
    B = MatrixXd::Identity(_n, _k);
    C = MatrixXd::Identity(_m, _k);
    P = MatrixXd::Identity(_k, _k);
    Q = MatrixXd::Identity(_k, _k);
    R = MatrixXd::Identity(_m, _m);
    transition_func = [this](TransitionVariables v)
    { 
      VectorXd a = A * v.X;
      if (v.U.size() > 0) a += B * v.U;
      return a * v.dt;
    };
    observation_func = [this](ObservationVariables v) { return C * v.X; };
  }

  virtual ~SequentialBayesianFilter() {}

  /// @brief get next state
  /// @return predicted state
  VectorXd predict() { return predict(VectorXd()); }

  /// @brief get next state
  /// @param u control input
  /// @return predicted state
  virtual VectorXd predict(const VectorXd& u) = 0;

  /// @brief correct current state
  /// @param y observation
  /// @return corrected state
  virtual VectorXd update(const VectorXd& y)  = 0;

};

////////////////////////////////////////////////////////////////

class ExtendedKalmanFilter final : public SequentialBayesianFilter
{
private:
  explicit ExtendedKalmanFilter(const ExtendedKalmanFilter&) : SequentialBayesianFilter(){}
  void operator=(const ExtendedKalmanFilter&){}

public:
  explicit ExtendedKalmanFilter(
    const VectorXd& x0, 
    const int& observation_vector_size, 
    const int& control_vector_size
  ) : SequentialBayesianFilter(x0, observation_vector_size, control_vector_size) {}

  // please update Jacobian (A: transition) in advance if you need.
  VectorXd predict(const VectorXd& u) override
  {
    if (dt <= 0) throw logic_error("Require: dt > 0");
    X = transition_func(TransitionVariables(X, u, dt));         // predict state estimate
    P = A * P * A.transpose() + Q;                              // predict estimate covariance
    P = (P + P.transpose()) * 0.5;
    return X;
  }

  // please update jacobian (C: observation) in advance if you need.
  VectorXd update(const VectorXd& y) override
  {
    VectorXd E = y - observation_func(ObservationVariables(X)); // innovation
    MatrixXd S = C * P * C.transpose() + R;                     // innovation covariance
    MatrixXd K = P * C.transpose() * S.inverse();               // optimal kalman gain
    X += K * E;                                                 // update state estimate
    P -= K * C * P;                                             // update estimate covariance
    P = (P + P.transpose()) * 0.5;
    return X;
  }

};

////////////////////////////////////////////////////////////////

class UnscentedKalmanFilter final : public SequentialBayesianFilter
{
private:
  explicit UnscentedKalmanFilter(const UnscentedKalmanFilter&) : SequentialBayesianFilter(){}
  void operator=(const UnscentedKalmanFilter&){}
  double _lambda;
  double _wi;
  double _ws0;
  double _wc0;
  vector<VectorXd> _sigmas;
  // put sigma points around the current estimate depending on the weight and P
  void update_sigmas()
  {
    for (int i = 0; i < P.rows(); i++)
      if (P(i, i) == 0) P(i, i) = 1e-9;
    MatrixXd s = _lambda * (MatrixXd)LLT<MatrixXd>(P).matrixL();
    _sigmas[0] = X;
    for (int i = 1; i <= _k; i++)
    {
      _sigmas[i]      = X + s.col(i - 1);
      _sigmas[_k + i] = X - s.col(i - 1);
    }
  }

public:
  explicit UnscentedKalmanFilter(
    const VectorXd& x0, 
    const int& observation_vector_size, 
    const int& control_vector_size,
    const double& alpha = 1e-3
  ) : SequentialBayesianFilter(x0, observation_vector_size, control_vector_size)
  {
    // determine weights
    double l = (pow(alpha, 2) - 1) * _k;
    _lambda = sqrt(_k + l);
    _wi = 1 / (2 * (_k + l));
    _ws0 = l / (l + _k);
    _wc0 = l / (l + _k) + 3 - pow(alpha, 2);
    _sigmas = vector<VectorXd>(2 * _k + 1);
  }

  VectorXd predict(const VectorXd& u) override
  {
    if (dt <= 0) throw logic_error("Require: dt > 0");
    update_sigmas();
    // move sigma points by the transition function
    // and estimate x depending on the weight
    VectorXd x = VectorXd::Zero(_k);
    for (int i = 0; i < (int)_sigmas.size(); i++)
    {
      _sigmas[i] = transition_func(TransitionVariables(_sigmas[i], u, dt));
      x += (i == 0 ? _ws0 : _wi) * _sigmas[i];
    }
    // evaluate prediction errors covariance
    MatrixXd p = MatrixXd::Zero(_k, _k);
    for (int i = 0; i < (int)_sigmas.size(); i++)
    {
      MatrixXd e(_sigmas[i] - x);
      p += (i == 0 ? _wc0 : _wi) * e * e.transpose();
    }
    // update X, P
    X = x;
    P = p + Q;
    P = (P + P.transpose()) * 0.5;
    return X;
  }

  VectorXd update(const VectorXd& y) override
  {
    update_sigmas();
    // estimate sigma points by the observation function
    // and estimate y depending on the weight
    vector<VectorXd> gammas(_sigmas.size());
    VectorXd ym = VectorXd::Zero(_m);
    for (int i = 0; i < (int)gammas.size(); i++)
    {
      gammas[i] = observation_func(ObservationVariables(_sigmas[i]));
      ym += (i == 0 ? _ws0 : _wi) * gammas[i];
    }
    // evaluate estimation errors covariance
    MatrixXd S = R;
    MatrixXd Cxy = MatrixXd::Zero(_k, _m);
    for (int i = 0; i < (int)gammas.size(); i++)
    {
      MatrixXd ey(gammas[i] - ym);
      MatrixXd ex(_sigmas[i] - X);
      S   += (i == 0 ? _wc0 : _wi) * ey * ey.transpose();
      Cxy += (i == 0 ? _wc0 : _wi) * ex * ey.transpose();
    }
    // calculate K and update X, P
    VectorXd E = y - ym;              // innovation
    MatrixXd K = Cxy * S.inverse();   // optimal kalman gain
    X += K * E;                       // update state estimate
    P -= K * S * K.transpose();       // update estimate covariance
    P = (P + P.transpose()) * 0.5;
    return X;
  }

};

////////////////////////////////////////////////////////////////

struct Particle final
{
  VectorXd state;
  double likelihood;
  explicit Particle() : Particle(VectorXd(), 0) {}
  explicit Particle(const VectorXd& state, const double& likelihood)
  {
    this->state = state;
    this->likelihood = likelihood;
  }
};

class ParticleFilter final : public SequentialBayesianFilter
{
private:
  explicit ParticleFilter(const ParticleFilter&) : SequentialBayesianFilter(){}
  void operator=(const ParticleFilter&){}

  void randomize(vector<Particle>& ps) const
  {
    random_device seed;
    default_random_engine engine(seed());
    for (int i = 0; i< (int)ps.size(); i++)
    {
      for (int j = 0; j < _k; j++)
      {
        VectorXd q = Q.row(j);
        for (int m = 0; m < (int)q.size(); m++)
        {
          if (q(m) == 0) continue;
          normal_distribution<> dist(0, sqrt(q(m)));
          ps[i].state(j) += dist(engine);
        }
      }
    }
  }

  VectorXd get_weighted_state(const vector<Particle>& ps) const
  {
    VectorXd state = VectorXd::Zero(_k);
    for (int i = 0; i < (int)ps.size(); i++)
      for (int j = 0; j < _k; j++)
        state(j) += ps[i].state(j) * ps[i].likelihood;
    return state;
  }

  void normalize_likelihood(vector<Particle>& ps) const
  {
    double sum = 0.0;
    for (int i = 0; i < (int)ps.size(); i++)
      sum += ps[i].likelihood;
    for (int i = 0; i < (int)ps.size(); i++)
      ps[i].likelihood = ps[i].likelihood / sum;
  }

public:
  vector<Particle> particles;
  function<vector<Particle> (vector<Particle>)> resampling_func;
  function<double (VectorXd)> get_likelihood_func;

  explicit ParticleFilter(
    const VectorXd& x0, 
    const int& observation_vector_size, 
    const int& control_vector_size,
    const int& n
  ) : SequentialBayesianFilter(x0, observation_vector_size, control_vector_size)
  {
    particles = vector<Particle>(n);
    for (int i = 0; i < n; i++)
      particles[i] = Particle(x0, 1.0 / n);
    randomize(particles);
    // systematic resampling
    resampling_func = [this](const vector<Particle>& ps)
    {
      vector<Particle> tmp;
      tmp.reserve(ps.size());
      double sum = 0.0;
      int i = 0;
      double scale = 1.0 / ps.size();
      for (double d = scale / 2; d < 1; d += scale)
      {
        while (sum + ps[i].likelihood < d)
        {
          sum += ps[i].likelihood;
          i++;
        }
        tmp.push_back(ps[i]);
      }
      return tmp;
    };
    // fitting a Gaussian distribution
    get_likelihood_func = [this](const VectorXd& e)
    {
      return exp(-0.5 * (e.transpose() * R.inverse() * e)(0, 0));
    };
  }

  VectorXd predict(const VectorXd& u) override
  {
    if (dt <= 0) throw logic_error("Require: dt > 0");
    vector<Particle> ps(particles);
    // predict next state
    for (int i = 0; i < (int)ps.size(); i++)
      ps[i].state = transition_func(TransitionVariables(ps[i].state, u, dt));
    randomize(ps);
    X = get_weighted_state(ps);
    particles = ps;
    return X;
  }

  VectorXd update(const VectorXd& y) override
  {
    vector<Particle> ps(particles);
    // update likelihood
    double sum = 0.0;
    double best = DBL_MAX;
    int index = 0;
    for (int i = 0; i < (int)ps.size(); i++)
    {
      VectorXd e = y - observation_func(ObservationVariables(ps[i].state));
      ps[i].likelihood = get_likelihood_func(e);
      sum += ps[i].likelihood;
      double norm = e.norm();
      if (norm < best)
      {
        index = i;
        best = norm;
      }
    }
    if (sum == 0) ps[index].likelihood = 1;
    normalize_likelihood(ps);
    X = get_weighted_state(ps);
    ps = resampling_func(ps);
    normalize_likelihood(ps);
    particles = ps;
    return X;
  }

};

#endif