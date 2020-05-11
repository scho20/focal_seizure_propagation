clear;
close all;
%% Time Parameters
simt = 500;         %simulation time (or 500)
dt = 1/1000;        %time step (or 1/1000)
T = 0:dt:simt;      %time series
lenT = length(T);
%% Parameters - Cell Type
% PYRAMIDAL EXCITATORY NEURON
n_half_e = -25;
EL_e = -80;
sa_e = 9;
% PARVALBUMINE INHIBITORY NEURON
n_half_i = -45;
EL_i = -78;
sa_i = 1;
%% Parameters - Synpase
C = 1;
E_Na = 60; E_K = -90;
g_L = 8; g_Na = 20; g_K = 10;
m_half = -20;
k_m = 15;
k_n = 5;
tau_n = 1;
% AMPA RECEPTOR
del_ampa = 2;
del_ampa = del_ampa/dt;
alpha_e = 25;               %amplitude factor of EPSP (mV)
beta_e = 0.3;               %decay factor of EPSP (sec^-1)
% GABA-A RECEPTOR
del_gaba = 5;
del_gaba = del_gaba/dt;
alpha_i = -3;               %amplitude factor of IPSP (mv)
beta_i = 0.1;               %decay factor of EPSP (sec^-1)
%% Transfer Function h(t)
% Alpha function adapted from van Rotterdam (1982)
he = alpha_e*beta_e*T.*exp(-beta_e*T);
hi = alpha_i*beta_i*T.*exp(-beta_i*T);

figure;
plot(T,he,'b'); hold on; plot(T,hi,'r');
xlabel('Time (ms)')
ylabel('Amplitude (\muA)')
title('Transfer Function: Alpha Function')
legend('Excitatory','Inhibitory')
%% Number of Neurons - per line
Ne_linear = 5;
Ni_linear = 5;
num_linear = Ne_linear + Ni_linear;

%figure;
%img = imread('../ffw_network.jpg');
%image(img)
%daspect([1,1,1]);
%set(gca,'visible','off');
%title('Basic Scheme of Single Linear Network')
%% Architecture of Neural Network
add_next_e = 5; %to mark E neurons with external input stimulation
add_next_i = 5; %to mark I neurons with external input stimulation
Ne = 5;
Ni = 5;
num_neurons = Ne+Ni;
num_lines = num_neurons/num_linear;
idx_e = 1:Ne;
idx_i = Ne+1:num_neurons;
if size(idx_e,1) ~= size(idx_i,1)
    error('Number of E and I neurons different; may have to change lattice structure')
end
%% External Current Injection

I_ext = zeros(num_neurons,lenT);
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
    
    % Random version: Set the input injection to scale * EPSP signal
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

    %I_inj = y*(I_ext_max/max(y));
    %I_inj_template = I_inj;
    
    % Ideal version: Without randomness
    pad_time = 15; % msec
    pad_length = pad_time*1000; % sec
    
    I_inj = zeros(1,lenT);
    I_inj(1:pad_length) = 0;
    I_inj(pad_length+1:end) = I_ext_max * linspace(0,1,length(pad_length+1:lenT));
    % If external input current is not padded with zeros, first dealy time
    % may be inconsistent with the rest of delay times.
    I_inj_template = I_inj;
    
    % Without padding zeros: I_inj = I_ext_max*linspace(0,1,lenT);
    
elseif choose_current == 2
    % [2] Create ramp current with exponential decay
    pad_time = 15;
    pad_length = pad_time*1000;
    
    ramp_duration = 0.4;
    decay_duration = 0.15/dt;

    I_inj = zeros(1,lenT);
    I_inj(1:pad_length) = 0;
    ramp_length = ceil(ramp_duration*lenT-pad_length);
    I_inj(pad_length+1:pad_length+ramp_length) = linspace(0,1,ramp_length);
    I_inj(pad_length+ramp_length+1:end) = exp(-(0:(lenT-pad_length-ramp_length-1))/(decay_duration*T(end)));
    I_inj_template = I_inj;
    I_inj = I_ext_max*I_inj;
    
    % Without padding zeros
    % I_inj = zeros(1,lenT);
    % ramp_length = ceil(ramp_duration*lenT);
    % I_inj(1:ramp_length) = linspace(0,1,ramp_length);
    % I_inj(ramp_length+1:end) = exp(-(0:(lenT-ramp_length-1))/(decay_duration*T(end)));
    % I_inj_template = I_inj;
    % I_inj = I_ext_max*I_inj;
elseif choose_current == 3
    % [3] Create current of box pulse
    box_start_fr = 0.1;
    box_end_fr = 0.8;
    box_start = ceil(box_start_fr*lenT);
    box_end = ceil(box_end_fr*lenT);
    
    I_inj = zeros(1,lenT);
    I_inj(box_start:box_end) = 1;
    I_inj_template = I_inj;
    I_inj = I_ext_max*I_inj;
elseif choose_current == 4
    % [4] Create current of Gaussian pulse
    T_gaus = -simt/2:dt:simt/2;
    mu = -30; % mean (positive: rightward / negative: leftward)
    bandwidth = 120^2; % standard deviation
    I_inj = exp(-(pi/bandwidth)*(T_gaus-mu).^2);
    I_inj_template = I_inj;
    I_inj = I_ext_max*I_inj;
end

figure;
plot(T,I_inj);
xlabel('Time (ms)')
ylabel('Current (pA)')
title('Injected External Current')

% (2) Current injection for stimulated neurons
idx_stim_e = 1:add_next_e:Ne; %index for stimulated E neurons
for i = idx_stim_e
    I_ext(i,:) = I_inj/sa_e; %inject current on E neurons
end
if choose_propagation == 1 %if feedfoward
    idx_stim_i = Ne+1:add_next_i:num_neurons; %index for stimulated I neurons
    for i = idx_stim_i
        I_ext(i,:) = I_inj/sa_i; %inject current on I neurons
    end
end

% (3) Add DC current
dc_app = 0;
I_ext = I_ext + dc_app;

% (4) Create surrounding currents
I_ext_nbr_max_E = 50; %scaling for E neurons
I_ext_nbr_max_I = 350; %scaling for I neurons

I_inj_nbr_E = I_ext_nbr_max_E * I_inj_template;
I_inj_nbr_I = I_ext_nbr_max_I * I_inj_template;
%% Initializing Simulation
array = zeros(num_neurons, lenT);
array_e = zeros(Ne, lenT);
array_i = zeros(Ni, lenT);
% (1) Initialize array for storage and access
x1 = array_e; y1 = array_e; z1 = array_e;
x2 = array_i; y2 = array_i; z2 = array_i;
n1 = array_e; n2 = array_i;
v1 = array_e; v2 = array_i;

dy1 = array_e; dy2 = array_i;
dz1 = array_e; dz2 = array_i;
dn1 = array_e; dn2 = array_i;
dv1 = array_e; dv2 = array_i;

n_inf1 = array_e; n_inf2 = array_i;
m_inf1 = array_e; m_inf2 = array_i;

MPe = zeros(Ne,lenT-1); MPi = zeros(Ni,lenT-1);
MPef = zeros(Ne,lenT-1); MPif = zeros(Ni,lenT-1);
dMPef = zeros(Ne,lenT-1); dMPif = zeros(Ni,lenT-1);

current_e = array_e;
current_i = array_i;
sur_current_e = zeros(Ne,lenT);
sur_current_i = zeros(Ni,lenT);
current_e_org = array_e;
current_i_org = array_i;
% (2) Assign Initial Values
z1(:,1)=0;
y1(:,1)=0;
z2(:,1)=0;
y2(:,1)=0;
% (3) Initialize spike detector
clear D;
%% High-Pass Filter for the Spike Detector
MPef(:,1) = 0;
MPif(:,1) = 0;
F = 0.2;
tauF = 1/(2*pi*F); %cutoff frequency F = 1/(2*pi*tauF)
%% Simulation Loop
% Calculating membrane potential of cells ('to') receiving afferents by
% accounting for cells ('from') that send efferents.

% (1) Initial conditions and variables for simulation
t = 0;
k = 1;
p = 1;

thr = 20;

v1(:,1) = -60;
v2(:,1) = -60;
n1(:,1) = 1/(1+exp((n_half_e-v1(1,1))/k_n));
n2(:,1) = 1/(1+exp((n_half_i-v2(1,1))/k_n));

k_prev_value = 1.5;
k_prev_init = k_prev_value/dt;
k_prev1 = zeros(Ne,1); k_prev1(:,1) = -k_prev_init;
k_prev2 = zeros(Ni,1); k_prev2(:,1) = -k_prev_init;

De = zeros(Ne,lenT);
Di = zeros(Ni,lenT);

detecter1 = 0; ind_e = cell(Ne,1);
detecter2 = 0; ind_i = cell(Ni,1);
loop1 = 1:Ne;
loop2 = 1:Ni;

% (2) Determining Synaptic Weights

we_e = 1; % E to E connection
% For E neurons, weight of 1~5 recommended; otherwise, easily saturates
we_i = 13; % E to I connection
% For I neurons, weight of 10~30 recommended; otherwise, fragile
wi_e = 1; % I to E connection
% For E neurons, weight of 0.8~1.2 recommended

% The reason why weights are selected to be [1, 15, 1] is because if we_i
% is bigger than 15, then the surrounding current has to be negative for
% first half theoretically.
% i.e. (External current - generated current = surrounding current)

% (3) Computing voltage potentials and propagating currents
fprintf('Starting simulation loop ...');
tic;
while (t<simt)
    
    % [1] High-Pass Filter & Spike Detection
    
    for i = 1:Ne
        MPe(i,k) = v1(i,k);
        if k>1
            dMPef(i,k) = (MPe(i,k)-MPe(i,k-1)) - MPef(i,k-1)*dt/tauF;
            MPef(i,k) = MPef(i,k-1) + dMPef(i,k);
        end
        if k>2 && (MPef(i,k-1) > thr)
            if MPef(i,k-1) >= MPef(i,k-2) && MPef(i,k-1) >= MPef(i,k) && ...
                    k+del_ampa < length(x1)
                % AMPA delayed activation
                if k - k_prev1(i,1) > k_prev_init
                    De(i,p) = k;
                    x1(i,k+del_ampa) = x1(i,k+del_ampa) + 1/dt;
                    k_prev1(i,1) = k;
                    p = p+1;
                end
            end
        end
    end
    
    for i = 1:Ni
        MPi(i,k) = v2(i,k);
        if k>1
            dMPif(i,k) = (MPi(i,k)-MPi(i,k-1)) - MPif(i,k-1)*dt/tauF;
            MPif(i,k) = MPif(i,k-1) + dMPif(i,k);
        end
        if k>2 && (MPif(i,k-1) > thr)
            if MPif(i,k-1) >= MPif(i,k-2) && MPif(i,k-1) >= MPif(i,k) && ...
                    k+del_gaba < length(x2)
                % GABA delayed activation
                if k - k_prev2(i,1) > k_prev_init
                    Di(i,p) = k;
                    x2(i,k+del_gaba) = x2(i,k+del_gaba) + 1/dt;
                    k_prev2(i,1) = k;
                    p = p+1;
                end
            end
        end
    end
        
    % [2] Compute Conductance and Synaptic Output for AMPA
    
    dy1(:,k) = z1(:,k);
    y1(:,k+1) = y1(:,k) + dy1(:,k)*dt;
    dz1(:,k) = alpha_e*beta_e*x1(:,k) - 2*beta_e*z1(:,k) - y1(:,k)*beta_e^2;
    z1(:,k+1) = z1(:,k) + dz1(:,k)*dt;
    
    m_inf1(:,k) = 1./(1+exp((m_half-v1(:,k))/k_m));
    n_inf1(:,k) = 1./(1+exp((n_half_e-v1(:,k))/k_n));
    dn1(:,k) = (n_inf1(:,k)-n1(:,k))/tau_n;
    n1(:,k+1) = n1(:,k) + dn1(:,k)*dt;
    
    % [3] Compute Conductance and Synaptic Output for GABA-A

    dy2(:,k) = z2(:,k);
    y2(:,k+1) = y2(:,k)+dy2(:,k)*dt;
    dz2(:,k) = alpha_i*beta_i*x2(:,k) - 2*beta_i*z2(:,k) - y2(:,k)*beta_i^2;
    z2(:,k+1) = z2(:,k) + dz2(:,k)*dt;
    
    m_inf2(:,k) = 1./(1+exp((m_half-v2(:,k))/k_m));
    n_inf2(:,k) = 1./(1+exp((n_half_i-v2(:,k))/k_n));
    dn2(:,k) = (n_inf2(:,k)-n2(:,k))/tau_n;
    n2(:,k+1) = n2(:,k) + dn2(:,k)*dt;
    
    % [4] Set Current
    
    % |A| Calculate Postsynaptic Currents
    if choose_propagation == 1 %if feedforward
        % For stimulated neurons
        for i = idx_stim_e
            current_e(i,k) = I_ext(i,k) + y2(i,k)*wi_e;
        end
        for i = idx_stim_i
            current_i(i-Ne,k) = I_ext(i,k);
        end
        % For propagating neurons
        for i = setdiff(2:Ne,idx_stim_e)
            current_e(i,k) = I_ext(i,k) + y1(i-1,k)*we_e + y2(i,k)*wi_e;
        end
        for i = setdiff(2:Ni,idx_stim_i-Ne)
            current_i(i,k) = I_ext(i+Ne,k) + y1(i-1,k)*we_i;
        end
    elseif choose_propagation == 2 %if feedback
        % For stimulated neurons
        for i = idx_stim_e
            current_e(i,k) = I_ext(i,k) + y2(i,k)*wi_e;
        end
        % For propagating neurons
        for i = setdiff(2:Ne,idx_stim_e)
            current_e(i,k) = I_ext(i,k) + y1(i-1,k)*we_e + y2(i,k)*wi_e;
        end
        for i = 1:Ni
            current_i(i,k) = I_ext(i+Ne,k) + y1(i,k)*we_i;
        end
    end
    
    % |B| Add presumptive surrounding currents
    
    if detecter1 == 0
        %ind_e = find(current_e(2,:),1,'first');
        for i = loop1
            if current_e(i,k) ~= 0
                ind_e{i} = k;
            end
            if ~isempty(ind_e{i}) == 1
                loop1 = setdiff(loop1,i);
            end
        end
        if ~isempty(ind_e{Ne})
            detecter1 = 1;
        end
    end
    
    for i = setdiff(1:Ne,idx_stim_e)
        if ~isempty(ind_e{i})
            if k >= ind_e{i}
                sur_current_e(i,k) = I_inj_nbr_E(1,k-ind_e{i}+1);
            end
        end
    end

    if detecter2 == 0
        %ind_i = find(current_i(2,:),1,'first');
        for i = loop2
            if current_i(i,k) ~= 0
                ind_i{i} = k;
            end
            if ~isempty(ind_e{i})
                loop2 = setdiff(loop2,i);
            end
        end
        if ~isempty(ind_i{Ni})
            detecter2 = 1;
        end
    end
    
    for i = setdiff(1:Ni,idx_stim_i-Ne)
        if ~isempty(ind_i{i})
            if k >= ind_i{i}
                sur_current_i(i,k) = I_inj_nbr_I(1,k-ind_i{i}+1);
            end
        end
    end
    
    current_e_org(:,k) = current_e(:,k);
    current_i_org(:,k) = current_i(:,k);
    
    for i = 1:Ne
        current_e(i,k) = current_e(i,k) + sur_current_e(i,k);
    end
    for i = 1:Ni
        current_i(i,k) = current_i(i,k) + sur_current_i(i,k);
    end
    
    % [5] Calculate Membrane Potentials
    for i = 1:Ne
        dv1(i,k) = (dt/C)*(current_e(i,k)-g_L*(v1(i,k)-EL_e)-g_Na*...
            m_inf1(i,k)*(v1(i,k)-E_Na)-g_K*n1(i,k)*(v1(i,k)-E_K));
        % The new membrane potential of the pyramidal cell
        v1(i,k+1) = v1(i,k) + dv1(i,k);
    end
    
    for i = 1:Ni
        dv2(i,k) = (dt/C)*(current_i(i,k)-g_L*(v2(i,k)-EL_i)-g_Na*...
            m_inf2(i,k)*(v2(i,k)-E_Na)-g_K*n2(i,k)*(v2(i,k)-E_K));
        % The new membrane potential of the parvalbumine cell
        v2(i,k+1) = v2(i,k) + dv2(i,k);
    end
    
    % [6] Update t & k
    t = t + dt;
    k = k + 1;
end
elapsed_time = toc;
fprintf('has taken %.3f seconds. \n', elapsed_time)
%% Plot
% Setting plot number marker
numplot = min(Ne, Ne_linear);

figure;
hold on
for i = 1:numplot
    plot(T,current_e_org(i,:));
end
xlabel('Time (ms)')
ylabel('Synaptic Current (pA)')
title('Original Current of the Ictal Network: Received by E Neurons')

figure;
hold on
for i = 1:numplot
    plot(T,current_i_org(i,:));
end
xlabel('Time (ms)')
ylabel('Synaptic Current (pA)')
title('Original Current of the Ictal Network: Received by I Neurons')

figure;
hold on
for i = 1:numplot
    plot(T,sur_current_e(i,:));
end
xlabel('Time (ms)')
ylabel('Synaptic Current (pA)')
title('Excitatory Artificial Current from Neighboring Neurons')

figure;
hold on
for i = 1:numplot
    plot(T,sur_current_i(i,:));
end
xlabel('Time (ms)')
ylabel('Synaptic Current (pA)')
title('Inhibitory Artificial Current from Neighboring Neurons')

figure;
hold on
lgnde = cell(1,numplot);
for i = 1:numplot
    plot(T,current_e(i,:));
    lgnde{1,i} = sprintf('Neuron #%d', i);
end
title('Total Current Received EPSP+IPSP: E Neurons')
xlabel('Time (ms)')
ylabel('Synaptic Current (pA)')
ylim([-inf max(max(current_e))+3])
legend(lgnde,'Location','southeastoutside')

figure;
hold on
lgndi = cell(1,numplot);
for i = 1:numplot
    plot(T,current_i(i,:));
    lgndi{1,i} = sprintf('Neuron #%d', i);
end
title('Total Current Received EPSP+IPSP: I Neurons')
xlabel('Time (ms)')
ylabel('Synaptic Current (pA)')
ylim([-inf max(max(current_i))+30])
legend(lgndi,'Location','southeastoutside')

figure;
for i = 1:numplot
    subplot(numplot,1,i);
    plot(T(1:length(MPe)), MPe(i,:), 'k');
    hold on
    plot(T(1:length(MPe)), y1(i,1:length(MPe)), 'b');
    He = ones(length(De),1)*max(MPe(i,:));
    plot(De(i,:)*dt,He,'.','Color',[0, 0.4470, 0.7410]);
    xlabel('Time (ms)')
    ylabel('MP (mV)')
    title(['Excitatory Neurons #',num2str(i)])
end

figure;
for i = 1:numplot
    subplot(numplot,1,i);
    plot(T(1:length(MPi)), MPi(i,:), 'k');
    hold on
    plot(T(1:length(MPi)), y2(i,1:length(MPi)), 'r');
    Hi = ones(length(Di),1)*max(MPi(i,:));
    plot(Di(i,:)*dt,Hi,'.','Color',[0.6350, 0.0780, 0.1840]);
    xlabel('Time (ms)')
    ylabel('MP (mV)')
    title(['Inhibitory Neurons #',num2str(i)])
end

fprintf('Temporary pause to check results. Press spacebar to continue. \n');
pause;
%% Propagation Delays
prop_del_e = zeros(Ne,1);
prop_del_i = zeros(Ni,1);
delay_time_e = zeros(Ne,1);
delay_time_i = zeros(Ni,1);
for i = 1:Ne
    prop_del_e(i,1) = find(De(i,:),1,'first');
end
for i = 1:Ni
    prop_del_i(i,1) = find(Di(i,:),1,'first');
end

conde = add_next_e*(1:num_lines);
condi = add_next_i*(1:num_lines);

for j = setdiff(1:Ne,conde)
    delay_time_e(j,1) = (De(j+1,prop_del_e(j+1,1)) - De(j,prop_del_e(j,1)))*dt;
end
delay_time_e = delay_time_e(delay_time_e~=0);% unit: msec
for j = setdiff(1:Ni,condi)
    delay_time_i(j,1) = (Di(j+1,prop_del_i(j+1,1)) - Di(j,prop_del_i(j,1)))*dt;
end
delay_time_i = delay_time_i(delay_time_i~=0); % unit: msec
%% Spike Counts
Spike_Counts = cell(num_neurons,1);
for i = 1:Ne
    store = De(i,:);
    Spike_Counts{i} = sort(store(store~=0)*dt/1000);
end
for i = 1:Ni
    store = Di(i,:);
    Spike_Counts{i+Ne} = sort(store(store~=0)*dt/1000);
end
%% Spike Estimation: Peristimulus Time Histogram (PSTH)
if ~all(cellfun(@(x) isvector(x) && isnumeric(x), Spike_Counts, 'uni', 1))
    error('Some neurons have not fired spikes.');
end

bin_size = 0.01;
smooth_size = 10.0;
edge_lim = [0, simt]*1e-3;

bin_num = ceil(abs(diff(edge_lim))/bin_size);
bin_edges = linspace(edge_lim(1),edge_lim(2),bin_num+1);
bin_centers = (bin_edges(1:end-1) + bin_edges(2:end))/2;

binned_psth = cellfun(@(x) histcounts(x(:),bin_edges)/bin_size, ...
    Spike_Counts, 'uni', 0); %averaged by bin size
smooth_psth = cellfun(@(x) smoothdata(x,'gaussian',smooth_size), ... 
    binned_psth, 'uni', 0);

% The command ['uni', 0 or false] results in the output return as a cell
% array; otherwise, output concatenates into a vector array.

PSTH = struct();
PSTH.binned = vertcat(binned_psth{:});
PSTH.smoothed = vertcat(smooth_psth{:});
PSTH.centers = bin_centers;

estimate_time = PSTH.centers;
estimate_rate = PSTH.smoothed;
%% Saved Changed Parameters
filename = 'simulation_info';

variable = struct();
variable.population.E = Ne;
variable.population.I = Ni;
variable.population.total = num_neurons;
if choose_current == 1
    variable.current_type = 'ramp';
elseif choose_current == 2
    variable.current_type = 'ramp_exp';
elseif choose_current == 3
    variable.current_type = 'box_pulse';
elseif choose_current == 4
    variable.current_type = 'gaussian_pulse';
end
variable.dc_current = dc_app;
variable.maxinjected = I_ext_max;
variable.weight.EtoE = we_e;
variable.weight.EtoI = we_i;
variable.weight.ItoE = wi_e;
variable.psth = PSTH;
save([filename '.mat'], 'variable');
%% Create Movie for Ictal Propagation
fprintf('Producing a video ...');
tic;

addpath('../') % for 'return_combination.m' and 'color2gradient.m' functions

% [1] Transform time steps into seconds for time series
T_sec = T/1000; % from ms to s

% [2] Set axes positions and legend location
% ax = [a, b, width, height] where a, b are Cartesian points of lower left
% corner
ax1_pos = [0.05, 0.15, 0.5, 0.75];
ax2_pos = [0.6, 0.65, 0.35, 0.25];
ax3_pos = [0.6, 0.35, 0.35, 0.25];
ax4_pos = [0.6, 0.05, 0.35, 0.25];
ax1_legend_loc = 'southoutside';

% [3] Initiate video writer
video_write = VideoWriter([filename '.avi']);
open(video_write);

title_style = {'FontSize',12,'FontWeight','normal'};

% [4] Generate lattice template
% Partly taken from 'set_pseudo_coordinate' & 'set_lattice_coordinate' 
% (https://github.com/tuanpham96/seizure_model)

lattice_distance = 2;

% |A| Build the unit lattice
%lattice_template = return_combination([-0.5,0.5], [-0.5,0.5]);
%lattice_template = vertcat(lattice_template, [0,0]); 
%index_template = [ones(4,1); -1];

lattice_template = [-0.5 0.5; 0 0];
index_template = [ones(1,1); -ones(1,1)]; % +1 = E, -1 = I

% |B| Recalculate number of neurons for template and final lattice
lattice_tplt_size = size(lattice_template, 1);
num_on_side = num_lines;
num_template = num_on_side*(num_linear/lattice_tplt_size);
new_num_neurons = num_template*lattice_tplt_size;

% |C| Major lattice to distribute the template
major_axis_height = sort((-(num_on_side-1):0) * lattice_distance, 'descend');
if Ne_linear == Ni_linear
    major_axis_width = (0:(Ne_linear-1)) * lattice_distance;
end
major_lattice = return_combination(major_axis_width, major_axis_height);
%(width, height) = (x,y); order is important

% |D| Final lattice and marking stimulated indices
final_lattice = arrayfun(@(i) lattice_template + major_lattice(i,:), ...
    1:num_template, 'uni', 0);
final_lattice = vertcat(final_lattice{:});
final_idx = repmat(index_template, [num_template,1]);

% |E| Reorganize indices and reset population indices
% Reorganize
[~,idx_sort] = sort(final_idx, 'descend');
final_idx = final_idx(idx_sort);
final_lattice = final_lattice(idx_sort, :);
% Reset
idx_e = sort(find(final_idx == +1), 'ascend');
idx_i = sort(find(final_idx == -1), 'ascend');
num_neurons = new_num_neurons;

% |F| Secure the starting coordinates
% Obtain pseudo coordinates
x_coord = final_lattice(:,1);
y_coord = final_lattice(:,2);
coordinates = struct('x', x_coord, 'y', y_coord);
% Get left coordinate
xleft = min(x_coord);
yleft = max(y_coord);
left_ind = find(x_coord == xleft & y_coord == yleft, 1);
left_neuron = struct('x', xleft, 'y', yleft, 'ind', left_ind);

% |G| Color Schemes
yellow = [0.9290, 0.6940, 0.1250];
maroon = [0.6350, 0.0780, 0.1840];
%purple = [0.4940, 0.1840, 0.5560];
%green = [0.4660, 0.6740, 0.1880];
%grey = [0.25, 0.25, 0.25];
%blue = [0, 0.4470, 0.7410];
color_opt = 'dual';
color = struct('E',yellow,'I',maroon,'option',color_opt);

% [5] Generate first frame
% |A| Visualize the neuronal population and stimulated neurons
% Partly taken from 'visualize_population' function. 
% For necessary reference, visit https://github.com/tuanpham96/seizure_model
spatial_pattern = 'LATTICE';
switch spatial_pattern
    case 'LATTICE'
        x_lat = coordinates.x;
        y_lat = coordinates.y;
        mark = 1/2;
end

figure; hold on;
graphics4legend = gobjects(1,3);
graphics4legend(1,1:2) = cellfun(@(pop,clr,nme) scatter(x_lat(pop), y_lat(pop), 150*mark, ...
    clr, 'filled', 'o', 'DisplayName', nme), ...
    {idx_e, idx_i}, {yellow, maroon}, {'EXC PYR neuron', 'INH PAR neuron'});
if ~isempty(idx_stim_e)
    graphics4legend(3,3) = scatter(x_lat(idx_stim_e), y_lat(idx_stim_e), 300*mark, ...
        'k', 'o', 'DisplayName', 'Stimulated Neuron');
end
graphics4legend = graphics4legend(isgraphics(graphics4legend));
legend(graphics4legend, 'Location', 'southoutside');
daspect([1,1,1]);

graphics4legend(3).DisplayName = 'Received external input';

% |B| Set axis for the first frame
set(gcf, 'units', 'normalized', 'position', [0,0,1,1], 'color', 'w');
ax1 = gca;
hold(ax1, 'on');
set(ax1, 'xcolor', 'none', 'ycolor', 'none', 'position', ax1_pos);
legend(ax1, graphics4legend, 'Location', ax1_legend_loc);

% |6| Min-Max Normalization for E and I Activities in PSTH
min_e = min(min(estimate_rate(idx_e,:)));
max_e = max(max(estimate_rate(idx_e,:)));
min_i = min(min(estimate_rate(idx_i,:)));
max_i = max(max(estimate_rate(idx_i,:)));

newmin = 0; newmax = 1;

estimate_rate_e = (estimate_rate(idx_e,:) - min_e) ./ (max_e - min_e);
% The minimum value in estimate_rate is mapped to 0, and the maximum value
% in estimate_rate is mapped to 1.
% The entire range of values of estimate_rate from its min to max are
% ampped to the range 0 to 1.
estimate_rate_e = (newmax - newmin) * estimate_rate_e + newmin;
% To transform (or normalize) the data into a particular range, then we can
% expand the general formula to:
% v' = (v-min)/(max-min) * (newmax-newmin) + newmin
% where newmin is the minimum and newmax is the maximum of the normalized
% dataset such that v = [min,max] -> v' = [newmin,newmax].
estimate_rate_i = (estimate_rate(idx_i,:) - min_i) ./ (max_i - min_i);
estimate_rate_i = (newmax - newmin) * estimate_rate_i + newmin;

estimate_current = I_inj_template;

% [7] Plotting Input Current, E Neuron Activity, & I Neuron Activity
axis_style = {'unit', 'normalized', 'visible', 'off'};
ax2 = axes(axis_style{:}, 'position', ax2_pos);
ax3 = axes(axis_style{:}, 'position', ax3_pos);
ax4 = axes(axis_style{:}, 'position', ax4_pos);
arrayfun(@(ax) hold(ax, 'on'), [ax1, ax2, ax3, ax4])

plot(ax2, T_sec, estimate_current, '-k', 'LineWidth', 1);
plot(ax3, estimate_time, estimate_rate_e, 'LineWidth', 1, 'Color', yellow);
plot(ax4, estimate_time, estimate_rate_i, 'LineWidth', 1, 'Color', maroon);

% [8] Final set up for video
% Overlay the first frame with absent activity
default_handle = scatter(ax1, x_lat, y_lat, 70*mark, 'w', 'filled', 'o');
title(ax1, 't = 0s', title_style{:});
% Vertical lines representing time
vert_time_lines = arrayfun(@(ax) plot(ax, [0,0], [0,1], ':k', ...
    'LineWidth', 0.7), [ax2, ax3, ax4]);
% Save to video
video_frame = getframe(gcf);
writeVideo(video_write, video_frame);

% [9] Transform discrete amplitude levels to discrete color contrasts
num_level = 100;

if strcmpi(color.option, 'dual') % compare strings
    cmap_e = flipud(color2gradient(color.E, num_level));
    cmap_i = flipud(color2gradient(color.I,num_level));
    sep_colors = true; % true = 1
end

if sep_colors == 1
    estimate_amp = estimate_rate;
    estimate_range = [min(estimate_rate(idx_e,:), [], 'all'), max(estimate_rate(idx_e,:), [], 'all')];
    estimate_amp(idx_e,:) = (estimate_rate(idx_e,:) - estimate_range(1))/abs(diff(estimate_range));
    estimate_range = [min(estimate_rate(idx_i,:), [], 'all'), max(estimate_rate(idx_i,:), [],'all')];
    estimate_amp(idx_i,:) = (estimate_rate(idx_i,:) - estimate_range(1))/abs(diff(estimate_range));
    estimate_level = round((num_level -1) * estimate_amp) + 1;
end

% [10] Plot and write frame for each time point
for i = 1:length(estimate_time)
    ti = estimate_time(i);
    ri = estimate_level(:,i);
    % Overlay with the current amplitude
    delete(default_handle);
    if i ==1 && sep_colors 
        sc_handle = gobjects(2,1);
    end
    
    if sep_colors
        sc_handle(1) = scatter(ax1, x_lat(idx_e), y_lat(idx_e), 120*mark, cmap_e(ri(idx_e),:), 'o', 'filled');
        sc_handle(2) = scatter(ax1, x_lat(idx_i), y_lat(idx_i), 120*mark, cmap_i(ri(idx_i),:), 'o', 'filled');
    end
    
    ti_sec = floor(ti);
    ti_ms = 1000*(ti-ti_sec);
    title(ax1, sprintf('Ictal Wave Propagation: Paroxysmal Depolarization Shift \nTime: t = %.0fs, %03.fms \n', ti_sec, ti_ms), title_style{:});
    legend(ax1, graphics4legend, 'Location', ax1_legend_loc);
    
    % Move vertical time line
    delete(vert_time_lines);
    time_lines = arrayfun(@(ax) plot(ax, ti*[1,1], [0,1], ':k', ...
        'LineWidth', 0.7), [ax2, ax3, ax4]);
    
    % Save to video
    video_frame = getframe(gcf);
    writeVideo(video_write, video_frame);
end

close(video_write);

elapsed_video = toc;
fprintf('has taken %.3f seconds. \n', elapsed_video);