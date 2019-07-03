% Mel Filter bank design

clc ; clear ; close

fmin = 20;
fmax = 20e3;

mmin = 1125*log(1+fmin/700); % min in mel
mmax = 1125*log(1+fmax/700); % max in mel

numF = 10; % Ten Filters.

numF = numF + 2;

m = linspace(mmin, mmax, numF);
f = 700.*(exp(m/1125)-1);
FB(1) = 0;

figure();
hold on;

for c = 1:(numF-2)
  FB(1) = f(c);
  FB(2) = f(c+1);
  FB(3) = f(c+2);
  plot(FB, [0,1,0]);
end

title('Mel Spaced Filter Bank');
xlabel('f (hz)');
ylabel('Mel Scale');

set(gca,'FontSize',20);

print('FA.eps');
