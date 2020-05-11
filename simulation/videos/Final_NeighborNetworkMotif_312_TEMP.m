clear;
close all;
%% Time Parameters
simt = 500;         %simulation time (or 500)
dt = 1/1000;        %time step (or 1/1000)
T = 0:dt:simt;      %time series
lenT = length(T);
%% Parameters - Cell Type
% PYRAMIDAL EXCITATORY NEURON
n_half_e_nbr = -25;
EL_e_nbr = -80;
sa_e = 9;
% PARVALBUMINE INHIBITORY NEURON
% Neighboring neurons as RS
n_half_i_nbr = -25; %or -45 for FS neuron
EL_i_nbr = -80; %or -78 for FS neuron
sa_i = 1;
%% Parameters - Synpase
C = 1;
E_Na = 60; E_K = -90;
g_L = 8; g_Na = 20; g_K = 10;
m_half = -20;
k_m = 15;
k_n = 5;
tau_n = 1;
% AMPA RECEPTOR - NEIGHBOR
del_ampa = 2;
del_ampa = del_ampa/dt;
alpha_e_nbr = 5;            %amplitude factor of EPSP (mV)
beta_e_nbr = 0.1;           %decay factor of EPSP (sec^-1)
% GABA-A RECEPTOR - NEIGHBOR
del_gaba = 5;
del_gaba = del_gaba/dt;
alpha_i_nbr = -3;           %amplitude factor of IPSP (mV)
beta_i_nbr = 0.1;           %decay factor of IPSP (sec^-1)
%% Transfer Function h(t)
% Alpha function adapted from van Rotterdam (1982)
he_nbr = alpha_e_nbr*beta_e_nbr*T.*exp(-beta_e_nbr*T);
hi_nbr = alpha_i_nbr*beta_i_nbr*T.*exp(-beta_i_nbr*T);

figure(2)
He_nbr = plot(T,he_nbr); hold on;
Hi_nbr = plot(T,hi_nbr);
grid on;
box off;
He_nbr.LineWidth = 1;
Hi_nbr.LineWidth = 1;
xlabel('Time (ms)');
ylabel('Amplitude (\muA)')
ylim([-2 10])
title('Transfer Function: Alpha Function (Neighbor)')
legend('Excitatory','Inhibitory')
%% Architecture of Neural Network Motif
% Number of neurons in the motif
Ne_nbr = 2; %excitatory neurons
Ni_nbr = 2; %inhibitory neurons
num_nbr = Ne_nbr + Ni_nbr; %total number of neurons
%% External Current Injection
I_ext_nbr = zeros(num_nbr,lenT);
I_ext_max = 800; %maximum current injection

% (1) Create current
choose_current = 2; % 1 - ramp current
                    % 2 - ramp current with exponential decay
                    % 3 - box pulse current
                    % 4 - Gaussian pulse current

choose_propagation = 1; % 1 - feedforward
                        % 2 - feedback

if choose_current == 1
    % [1] Creat ramp current
    
    % Ideal version w/o randomness
    pad_time = 15; % msec
    pad_length = pad_time*1000; % sec
    
    %%%%%%%%%%%%%%%%%%%% FOR NEIGHBOR %%%%%%%%%%%%%%%%%%%%
    I_inj_nbr = zeros(1,lenT);
    I_inj_nbr(1:pad_length) = 0;
    I_inj_nbr(pad_length+1:end) = I_ext_max * linspace(0,1,length(pad_length+1:lenT));
    % If external input current is not padded with zeros, first dealy time
    % may be inconsistent with the rest of delay times.
    I_inj_template_nbr = I_inj_nbr;
    
    % Without padding zeros: I_inj_nbr = I_ext_max*linspace(0,1,lenT);
    
    %%%%%%%%%%%%%%%%%%%%%%%%% Randomized Current %%%%%%%%%%%%%%%%%%%%%%%%%
    %S_test = (I_ext_max/simt)*T;
    %S_in=zeros(1,length(T));          % initialize the input spike train
    %for k=1:length(T)
    %    tst=rand(1)*I_ext_max;
    %    if tst < S_test(k)*0.7        % arbitrary multiplyer to control spiking rate
    %        S_in(k)=1;                % insert a 'random' input spike
    %    end
    %end
    
    %z = zeros(1,lenT);
    %y = zeros(1,lenT);
    %z(1)=0;
    %y(1)=0;

    %for k=1:length(T)-1
    %    dy=z(k);
    %    y(k+1)=y(k)+dy*dt;
    %    dz=alpha_e*beta_e*S_in(k)-2*beta_e*z(k)-y(k)*beta_e^2;
    %    z(k+1)=z(k)+dz*dt;
    %end
    
    % Random version: Set the input inject to scale * EPSP signal
    % I_inj = y*(I_ext_max/max(y));
    % I_inj_template = I_inj;
    
elseif choose_current == 2
    % [2] Create ramp current with exponential decay
    pad_time = 15;
    pad_length = pad_time*1000;
    
    ramp_duration = 0.4;
    decay_duration = 0.15/dt;
    
    %%%%%%%%%%%%%%%%%%%% FOR NEIGHBOR %%%%%%%%%%%%%%%%%%%%%%%%%%%%
    I_inj_nbr = zeros(1,lenT);
    I_inj_nbr(1:pad_length) = 0;
    ramp_length = ceil(ramp_duration*lenT-pad_length);
    I_inj_nbr(pad_length+1:pad_length+ramp_length) = linspace(0,1,ramp_length);
    I_inj_nbr(pad_length+ramp_length+1:end) = exp(-(0:(lenT-pad_length-ramp_length-1))/(decay_duration*T(end)));
    
    I_inj_template_nbr = I_inj_nbr;
    I_inj_nbr = I_ext_max*I_inj_nbr;
    
    % Without padding zeros
    % I_inj_nbr = zeros(1,lenT);
    % ramp_length = ceil(ramp_duration*lenT);
    % I_inj_nbr(1:ramp_length) = linspace(0,1,ramp_length);
    % I_inj_nbr(ramp_length+1:end) = exp(-(0:(lenT-ramp_length-1))/(decay_duration*T(end)));
    % I_inj_template_nbr = I_inj_nbr;
    % I_inj_nbr = I_ext_max*I_inj_nbr;
    
elseif choose_current == 3
    % [3] Create current of box pulse
    box_start_fr = 0.1;
    box_end_fr = 0.8;
    box_start = ceil(box_start_fr*lenT);
    box_end = ceil(box_end_fr*lenT);
    
    I_inj_nbr = zeros(1,lenT);
    I_inj_nbr(box_start:box_end) = 1;
    I_inj_template_nbr = I_inj_nbr;
    I_inj_nbr = I_ext_max*I_inj_nbr;
elseif choose_current == 4
    % [4] Create current of Gaussian pulse
    T_gaus = -simt/2:dt:simt/2;
    mu = -30; % mean (positive: rightward / negative: leftward)
    bandwidth = 120^2; % standard deviation
    I_inj_nbr = exp(-(pi/bandwidth)*(T_gaus-mu).^2);
    I_inj_template_nbr = I_inj_nbr;
    I_inj_nbr = I_ext_max*I_inj_nbr;
end

figure(4)
plot(T,I_inj_nbr);
xlabel('Time (ms)')
ylabel('Current (pA)')
title('Injected External Current')

% (2) Current injection for stimulated neurons
idx_stim_e_nbr = 1; %to mark stimulated E neurons
I_ext_nbr(idx_stim_e_nbr,:) = I_inj_nbr/sa_e; %inject current on E neuron


if choose_propagation == 1 %if feedfoward
    idx_stim_i_nbr = 1+Ne_nbr; %to mark stimulated I neurons
    I_ext_nbr(idx_stim_i_nbr,:) = I_inj_nbr/sa_i; %inject current on I neurons
end

% (3) Add DC current
dc_app = 0;
I_ext_nbr = I_ext_nbr + dc_app;
%% Initializing Simulation
% (1) Initialize array for storage and access
% For neighboring neurons
array_nbr = zeros(num_nbr, lenT);
array_e_nbr = zeros(Ne_nbr, lenT);
array_i_nbr = zeros(Ni_nbr, lenT);

[x1_nbr, y1_nbr, z1_nbr, n1_nbr, v1_nbr] = deal(array_e_nbr);
[x2_nbr, y2_nbr, z2_nbr, n2_nbr, v2_nbr] = deal(array_i_nbr);

[dy1_nbr, dz1_nbr, dn1_nbr, dv1_nbr] = deal(array_e_nbr);
[dy2_nbr, dz2_nbr, dn2_nbr, dv2_nbr] = deal(array_i_nbr);

[n_inf1_nbr, m_inf1_nbr] = deal(array_e_nbr);
[n_inf2_nbr, m_inf2_nbr] = deal(array_i_nbr);

MPe_nbr = zeros(Ne_nbr,lenT-1); MPi_nbr = zeros(Ni_nbr,lenT-1);
MPef_nbr = zeros(Ne_nbr,lenT-1); MPif_nbr = zeros(Ni_nbr,lenT-1);
dMPef_nbr = zeros(Ne_nbr,lenT-1); dMPif_nbr = zeros(Ni_nbr,lenT-1);

[sur_current_e] = deal(array_e_nbr);
[sur_current_i] = deal(array_i_nbr);
% (2) Assign Initial Values
z1_nbr(:,1)=0;
y1_nbr(:,1)=0;
z2_nbr(:,1)=0;
y2_nbr(:,1)=0;
% (3) Initialize spike detector
clear D;
%% High-Pass Filter for the Spike Detector
MPef_nbr(:,1) = 0;
MPif_nbr(:,1) = 0;
F = 0.2;
tauF = 1/(2*pi*F); %cutoff frequency F = 1/(2*pi*tauF)
%% Simulation Loop
% Calculating membrane potential of cells ('to') receiving afferents by
% accounting for cells ('from') that send efferents.

% (1) Initial conditions and variables for simulation
t = 0;
k = 1;
p_nbr = 1;

thr = 20;
k_prev_value = 1.5;
k_prev_init = k_prev_value/dt;

% For neighboring neurons
v1_nbr(:,1) = -60;
v2_nbr(:,1) = -60;

n1_nbr(:,1) = 1/(1+exp((n_half_e_nbr-v1_nbr(1,1))/k_n));
n2_nbr(:,1) = 1/(1+exp((n_half_i_nbr-v2_nbr(1,1))/k_n));

k_prev1_nbr = zeros(Ne_nbr,1); k_prev1_nbr(:,1) = -k_prev_init;
k_prev2_nbr = zeros(Ni_nbr,1); k_prev2_nbr(:,1) = -k_prev_init;

De_nbr = zeros(Ne_nbr,lenT);
Di_nbr = zeros(Ni_nbr,lenT);
% (2) Determining Synaptic Weights
we_e_nbr = 1; % E to E connection
% For E neurons, weight of 1~5 recommended; otherwise, easily saturates
we_i_nbr = 15; % E to I connection
% For I neurons, weight of 10~30 recommended; otherwise, fragile
wi_e_nbr = 1; % I to E connection
% For E neurons, weight of 0.8~1.2 recommended

% (3) Computing voltage potentials and propagating currents
fprintf('Starting simulation loop ...');
tic;
while (t<simt)
    
    % [1] High-Pass Filter & Spike Detection  
    % FOR NEIGHBORING NEURONS
    for i = 1:Ne_nbr
        MPe_nbr(i,k) = v1_nbr(i,k);
        if k>1
            dMPef_nbr(i,k) = (MPe_nbr(i,k)-MPe_nbr(i,k-1)) - MPef_nbr(i,k-1)*dt/tauF;
            MPef_nbr(i,k) = MPef_nbr(i,k-1) + dMPef_nbr(i,k);
        end
        if k>2 && (MPef_nbr(i,k-1) > thr)
            if MPef_nbr(i,k-1) >= MPef_nbr(i,k-2) && MPef_nbr(i,k-1) >= MPef_nbr(i,k) && ...
                    k+del_ampa < length(x1_nbr)
                % AMPA delayed activation
                if k - k_prev1_nbr(i,1) > k_prev_init
                    De_nbr(i,p_nbr) = k;
                    x1_nbr(i,k+del_ampa) = x1_nbr(i,k+del_ampa) + 1/dt;
                    k_prev1_nbr(i,1) = k;
                    p_nbr = p_nbr+1;
                end
            end
        end
    end
    
    for i = 1:Ni_nbr
        MPi_nbr(i,k) = v2_nbr(i,k);
        if k>1
            dMPif_nbr(i,k) = (MPi_nbr(i,k)-MPi_nbr(i,k-1)) - MPif_nbr(i,k-1)*dt/tauF;
            MPif_nbr(i,k) = MPif_nbr(i,k-1) + dMPif_nbr(i,k);
        end
        if k>2 && (MPif_nbr(i,k-1) > thr)
            if MPif_nbr(i,k-1) >= MPif_nbr(i,k-2) && MPif_nbr(i,k-1) >= MPif_nbr(i,k) && ...
                    k+del_gaba < length(x2_nbr)
                % GABA delayed activation
                if k - k_prev2_nbr(i,1) > k_prev_init
                    Di_nbr(i,p_nbr) = k;
                    x2_nbr(i,k+del_gaba) = x2_nbr(i,k+del_gaba) + 1/dt;
                    k_prev2_nbr(i,1) = k;
                    p_nbr = p_nbr+1;
                end
            end
        end
    end
    
    % [2] Compute Conductance and Synaptic Output for AMPA
    % FOR NEIGHBORING NEURONS
    dy1_nbr(:,k) = z1_nbr(:,k);
    y1_nbr(:,k+1) = y1_nbr(:,k) + dy1_nbr(:,k)*dt;
    dz1_nbr(:,k) = alpha_e_nbr*beta_e_nbr*x1_nbr(:,k) - 2*beta_e_nbr*z1_nbr(:,k) - y1_nbr(:,k)*beta_e_nbr^2;
    z1_nbr(:,k+1) = z1_nbr(:,k) + dz1_nbr(:,k)*dt;
    
    m_inf1_nbr(:,k) = 1./(1+exp((m_half-v1_nbr(:,k))/k_m));
    n_inf1_nbr(:,k) = 1./(1+exp((n_half_e_nbr-v1_nbr(:,k))/k_n));
    dn1_nbr(:,k) = (n_inf1_nbr(:,k)-n1_nbr(:,k))/tau_n;
    n1_nbr(:,k+1) = n1_nbr(:,k) + dn1_nbr(:,k)*dt;
    
    % [3] Compute Conductance and Synaptic Output for GABA-A
    % FOR NEIGHBORING NEURONS
    dy2_nbr(:,k) = z2_nbr(:,k);
    y2_nbr(:,k+1) = y2_nbr(:,k)+dy2_nbr(:,k)*dt;
    dz2_nbr(:,k) = alpha_i_nbr*beta_i_nbr*x2_nbr(:,k) - 2*beta_i_nbr*z2_nbr(:,k) - y2_nbr(:,k)*beta_i_nbr^2;
    z2_nbr(:,k+1) = z2_nbr(:,k) + dz2_nbr(:,k)*dt;
    
    m_inf2_nbr(:,k) = 1./(1+exp((m_half-v2_nbr(:,k))/k_m));
    n_inf2_nbr(:,k) = 1./(1+exp((n_half_i_nbr-v2_nbr(:,k))/k_n));
    dn2_nbr(:,k) = (n_inf2_nbr(:,k)-n2_nbr(:,k))/tau_n;
    n2_nbr(:,k+1) = n2_nbr(:,k) + dn2_nbr(:,k)*dt;
    
    % [4] Set Current
    % |A| Calculate postsynaptic surrounding currents from neighboring neurons
    if choose_propagation == 1 %if feedforward
        % For stimulated neurons
        for i = idx_stim_e_nbr
            sur_current_e(i,k) = I_ext_nbr(i,k) + y2_nbr(i,k)*wi_e_nbr;
        end
        for i = idx_stim_i_nbr
            sur_current_i(i-Ne_nbr,k) = I_ext_nbr(i,k);
        end
        % For propagatin neurons
        for i = setdiff(2:Ne_nbr,idx_stim_e_nbr)
            sur_current_e(i,k) = I_ext_nbr(i,k) +y1_nbr(i-1,k)*we_e_nbr + y2_nbr(i,k)*wi_e_nbr;
        end
        for i = setdiff(2:Ni_nbr,idx_stim_i_nbr-Ne_nbr)
            sur_current_i(i,k) = I_ext_nbr(i+Ne_nbr,k) + y1_nbr(i-1,k)*we_i_nbr;
        end
    end
    
    % [5] Calculate Membrane Potentials
    for i = 1:Ne_nbr
        dv1_nbr(i,k) = (dt/C)*(sur_current_e(i,k)-g_L*(v1_nbr(i,k)-EL_e_nbr)-g_Na*...
            m_inf1_nbr(i,k)*(v1_nbr(i,k)-E_Na)-g_K*n1_nbr(i,k)*(v1_nbr(i,k)-E_K));
        % The new membrane potential of the pyramidal cell
        v1_nbr(i,k+1) = v1_nbr(i,k) + dv1_nbr(i,k);
    end
    
    for i = 1:Ni_nbr
        dv2_nbr(i,k) = (dt/C)*(sur_current_i(i,k)-g_L*(v2_nbr(i,k)-EL_i_nbr)-g_Na*...
            m_inf2_nbr(i,k)*(v2_nbr(i,k)-E_Na)-g_K*n2_nbr(i,k)*(v2_nbr(i,k)-E_K));
        % The new membrane potential of the parvalbumine cell
        v2_nbr(i,k+1) = v2_nbr(i,k) + dv2_nbr(i,k);
    end
    
    % [6] Update t & k
    t = t + dt;
    k = k + 1;
end
elapsed_time = toc;
fprintf('has taken %.3f seconds. \n', elapsed_time)
%% Plot
% Setting plot number marker
figure;
hold on
lgnde = cell(1,Ne_nbr);
for i = 1:Ne_nbr
    plot(T,sur_current_e(i,:));
    lgnde{1,i} = sprintf('Neuron #%d', i);
end
xlabel('Time (ms)')
title('Neighboring Current: E Neuron')
legend(lgnde,'Location','northeast')

figure;
hold on
lgndi = cell(1,Ni_nbr);
for i = 1:Ni_nbr
    plot(T,sur_current_i(i,:));
    lgndi{1,i} = sprintf('Neuron #%d', i);
end
xlabel('Time (ms')
title('Neighboring Current: I Neuorn')
legend(lgndi,'Location','northeast')

figure;
for i = 1:Ne_nbr
    subplot(Ne_nbr,1,i);
    plot(T(1:length(MPe_nbr)), MPe_nbr(i,:), 'k');
    hold on
    plot(T(1:length(MPe_nbr)), y1_nbr(i,1:length(MPe_nbr)), 'b');
    He_nbr = ones(length(De_nbr),1)*max(MPe_nbr(i,:));
    plot(De_nbr(i,:)*dt,He_nbr,'.');
    xlabel('Time (ms)')
    ylabel('MP (mV)')
    title(['Excitatory Neurons #',num2str(i)])
end

figure;
for i = 1:Ni_nbr
    subplot(Ni_nbr,1,i);
    plot(T(1:length(MPi_nbr)), MPi_nbr(i,:), 'k');
    hold on
    plot(T(1:length(MPi_nbr)), y2_nbr(i,1:length(MPi_nbr)), 'r');
    Hi_nbr = ones(length(Di_nbr),1)*max(MPi_nbr(i,:));
    plot(Di_nbr(i,:)*dt,Hi_nbr,'.');
    xlabel('Time (ms)')
    ylabel('MP (mV)')
    title(['Inhibitory Neurons #',num2str(i)])
end
