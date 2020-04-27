#Attribution: Adapted From HASAN POONAWALA slam_simple_point.jl
function linearizePointModel(xTrue,Rvm,Rwm,yk,lmarks)
    nl = size(lmarks,1)
    # Initialize matrices
    A = zeros(3+3*nl,3+3*nl)
    B = zeros(3+3*nl,3)
    C = zeros(3*nl,3+3*nl)
    Rv = zeros(3*nl,3*nl)
    Rw = zeros(3+3*nl,3+3*nl)

    # Start populating
    A[1,1] = 1.0;
    A[1,2] = 0.0;
    A[1,3] = 0.0;
    A[2,1] = 0.0;
    A[2,2] = 1.0;
    A[2,3] = 0.0;
    A[3,1] = 0.0;
    A[3,2] = 0.0;
    A[3,3] = 1.0;
    
    B[1,1] = 1.0;
    B[2,2] = 1.0;
    B[3,3] = 1.0;

    Rw[1:3,1:3,1:3] = Rwm
    Rv[1:3,1:3,1:3] = Rvm

        Rw[1:3,1:3,1:3] = Rwm
    Rv[1:3,1:3,1:3] = Rvm

    for i in 1:nl
        A[3+3*i-2,3+3*i-2] = 1.0;
        A[3+3*i-1,3+3*i-1] = 1.0;
        A[3+3*i  ,3+3*i] = 1.0;
        Rv[(3*i-1):(3*i),(3*i-1):(3*i),(3*i-1):(3*i)] = Rvm;
        #Rw[(2+2*i-1):(2+2*i),(2+2*i-1):(2+2*i)] = 0.1*Rwm;
        C[3*i-2,1] = -1.0;
        C[3*i-1,2] = -1.0;
        C[3*i  ,3] = -1.0;
        C[3*i-2,3+3*i-2] = 1.0;
        C[3*i-1,3+3*i-1] = 1.0;
        C[3*i  ,3+3*i  ] = 1.0;
    end
    mysys = LTIsys(A,B,C,Rw,Rv)
    return mysys
end


# Start
Rv = diagm([0.003;0.005;0.004]) # range, bearing measurement noise
Rw = diagm([0.4;0.5;0.2]) # x, y, z  process noise
lmarks = [1.0 2.0 3.0]
nl = size(lmarks,1)
gr()
plot([1.0],[1.0],[1.0])

# initialize robot
global mu = zeros(3+3*nl);
global Sigma = diagm(15*ones(3+3*nl));
global xTrue = [4.0;4.0;4.0]

# Get model
mysys = linearizePointModel(xTrue,Rv,Rw,0.0,lmarks);

for i in 1:1000
    global mu
    global Sigma
    global xTrue
    #v = rand(1)[1]*0.5 - xTrue[1];
    #w = randn(1)[1]*0.5- xTrue[2];
    #uk = [v;w];
    #xTrue = mysys.A[1:2,1:2]*xTrue + uk + 1.0*Rw*randn(2,1);
    yk = zeros(3);
    yk[1] = lmarks[1,1] - xTrue[1];
    yk[2] = lmarks[1,2] - xTrue[2];
    yk[3] = lmarks[1,3] - xTrue[3];
    yk = yk+mysys.Rv*randn(3*nl,1)
    mu,Sigma = filter_update(mysys,mu,Sigma,uk,yk);
  
  
  end
