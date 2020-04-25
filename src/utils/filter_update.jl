#Attribution: Hasan Poonawala
function filter_update(sys,mu,Sigma,uk,yk)
    # x_{t+1} = A x_t + B u_t + w_t # Rw
    # y_{t} = C x_t #Rv
    A_d = sys.A;B = sys.B;C = sys.C;
    Rv = sys.Rv; Rw = sys.Rw;

    # Predict the state distribution after moving under uk
    mu_pred = A_d*mu + B*uk;
    Sigma_pred = A_d*(Sigma)*transpose(A_d)+ Rw;

    # Compute the Kalman gain
    Sk = C*Sigma_pred*transpose(C) + Rv
    Kk = Sigma_pred*transpose(C)*inv(Sk);
    M = diagm(ones(size(C,2)))
    Gainmat = M - Kk*C;


    # Update the distribution using measurement
    mu = mu_pred + Kk*(yk - C*mu_pred);
    Sigma =Gainmat*Sigma_pred;

    return mu, Sigma
end

#function add_plot_ellipse(c1,c2,Sigma,colo)
#  pts = Plots.partialcircle(0,2π,100,0.1)
#  x, y = Plots.unzip(pts)
#  p = [x';y'];
#  #p = [cos(theta) -sin(theta);sin(theta) cos(theta)]*[a 0.0; 0.0 b]*p + [c1;c2]*ones(1,length(x))
#  p = sqrt(3*Sigma)*p + [c1;c2]*ones(1,length(x))
#  x1 = p[1,:]
#  y1 = p[2,:]
#  if colo == 1
#  plot!(Shape(x1,y1), c=:blue,legend=:none,fillalpha=0.5)
#  end
#  if colo == 0
#  plot!(Shape(x1,y1), c=:red,legend=:none,fillalpha=0.5)
#  end
#end
