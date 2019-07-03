f = 0 : 0.1 : 20e3;
M = 1125*log(1+f./700);
plot(f, M);

title('The Mel Scale');
xlabel('f (hz)');
ylabel('Mel Scale');

set(gca,'FontSize',20);

print('mel.eps');
