# Binary classification as a phase separation process
Python code for the paper <b>Binary classification as a phase separation process</b>, by <b> <a href=https://sites.google.com/view/rafaelmonteiro-math/home>Rafael Monteiro </a></b>.

In this github you will the code for used in the paper and some examples of how to use/apply the model.

### Available Data and verifiability:
The dataset used in the paper is public: it is known as the <a href=http://yann.lecun.com/exdb/mnist/> MNIST database </a>.

With regards to trained parameters, posting all of them is impossible due to their size (almost 5gb). Therefore I'm making only the following trained models available:


<ol>
  <li>Repeat-1-Jump, Nt = 4, N_ptt = 196, Non-Diffusive, subordinated.</li>
  <li>Repeat-Nt-Jump, Nt = 4, N_ptt = 196, Non-Diffusive, subordinated.</li>
<li>Repeat-1-Jump, Nt = 4,N_ptt = 196, Non-Diffusive, non-subordinated.</li>
<li>Repeat-Nt-Jump, Nt = 4,N_ptt = 196, Non-Diffusive, non-subordinated.</li>
<li>Repeat-1-Jump, Nt = 4, N_ptt = 196, Diffusive (Neumann BCs).</li>
<li>Repeat-Nt-Jump, Nt = 4, N_ptt = 196, Diffusive  (Neumann BCs).</li>
<li>Repeat-1-Jump, Nt = 4, N_ptt = 196, Diffusive  (Periodic).</li>
<li>Repeat-Nt-Jump, Nt = 4, N_ptt = 196, Diffusive  (Periodic).</li>
</ol>
I'm glad to send them over for anyone who needs them while I cannot find a storage. Nevertheless, as soon as I find a storage to them I'll post the link here.

### Remarks:
All of them have been trained on clusters with 12 or 22 processors.

As I have proved in the paper, it is possible to parallelize the inviscid model, but it has not been implemented; you are welcome to do that if you want. 




<br>
--------------------------------------------------
</br>
<ul>
<li>Rafael Monteiro</li>
<li>Mathematics for Advanced Materials - Open Innovation Laboratory, Japan</li>
<li>PhD Applied Mathematics</li>
<li>https://sites.google.com/view/rafaelmonteiro-math/home</li>
<li>email:<a href=rafael.a.monteiro.math@gmail.com>rafael.a.monteiro.math  AT gmail.com</a>, or <a href=monteirodasilva-rafael@aist.go.jp>monteirodasilva-rafael AT aist.go.jp</a></li>
</ul>
