import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from torch import Tensor


def min_max_scale_each_sample(data : Tensor):
    min_val = data.min(dim=1, keepdim=True)[0]
    max_val = data.max(dim=1, keepdim=True)[0]
    scaled_data = (data-min_val)/(max_val - min_val)
    
    return scaled_data, min_val, max_val

def inverse_min_max_scale(scaled_data : Tensor,
                          min_val : Tensor, 
                          max_val : Tensor):
    origin_data = scaled_data*(max_val-min_val)+ min_val

    return origin_data
  
def visualization (ori_data, generated_data, analysis):
  """Using PCA or tSNE for generated and original data visualization.
  
  Args:
    - ori_data: original data
    - generated_data: generated synthetic data
    - analysis: tsne or pca
  """  
  # Analysis sample size (for faster computation)
  anal_sample_no = min([1000, len(ori_data)])
  idx = np.random.permutation(len(ori_data))[:anal_sample_no]
    
  # Data preprocessing
  ori_data = np.asarray(ori_data)
  generated_data = np.asarray(generated_data)  
  
  ori_data = ori_data[idx]
  generated_data = generated_data[idx]
  
  no, seq_len, dim = ori_data.shape  
  
  for i in range(anal_sample_no):
    if (i == 0):
      prep_data = np.reshape(np.mean(ori_data[0,:,:], 1), [1,seq_len])
      prep_data_hat = np.reshape(np.mean(generated_data[0,:,:],1), [1,seq_len])
    else:
      prep_data = np.concatenate((prep_data, 
                                  np.reshape(np.mean(ori_data[i,:,:],1), [1,seq_len])))
      prep_data_hat = np.concatenate((prep_data_hat, 
                                      np.reshape(np.mean(generated_data[i,:,:],1), [1,seq_len])))
    
  # Visualization parameter        
  colors = ["red" for i in range(anal_sample_no)] + ["blue" for i in range(anal_sample_no)]    
    
  if analysis == 'pca':
    # PCA Analysis
    pca = PCA(n_components = 2)
    pca.fit(prep_data)
    pca_results = pca.transform(prep_data)
    pca_hat_results = pca.transform(prep_data_hat)
    
    # Plotting
    f, ax = plt.subplots(1)    
    plt.scatter(pca_results[:,0], pca_results[:,1],
                c = colors[:anal_sample_no], alpha = 0.2, label = "Original")
    plt.scatter(pca_hat_results[:,0], pca_hat_results[:,1], 
                c = colors[anal_sample_no:], alpha = 0.2, label = "Synthetic")
  
    ax.legend()  
    plt.title('PCA plot')
    plt.xlabel('x-pca')
    plt.ylabel('y_pca')
    plt.savefig("pca_test.png")
    # plt.show()
    
  elif analysis == 'tsne':
    
    # Do t-SNE Analysis together       
    prep_data_final = np.concatenate((prep_data, prep_data_hat), axis = 0)
    
    # TSNE anlaysis
    tsne = TSNE(n_components = 2, verbose = 1, perplexity = 40, n_iter = 300)
    tsne_results = tsne.fit_transform(prep_data_final)
      
    # Plotting
    f, ax = plt.subplots(1)
      
    plt.scatter(tsne_results[:anal_sample_no,0], tsne_results[:anal_sample_no,1], 
                c = colors[:anal_sample_no], alpha = 0.2, label = "Original")
    plt.scatter(tsne_results[anal_sample_no:,0], tsne_results[anal_sample_no:,1], 
                c = colors[anal_sample_no:], alpha = 0.2, label = "Synthetic")
  
    ax.legend()
      
    plt.title('t-SNE plot')
    plt.xlabel('x-tsne')
    plt.ylabel('y_tsne')
    plt.savefig("sne_test.png")

    # plt.show()

def discriminative_score(num_batches, G, D, S, E, dataloader, batch_size, dim, seq_len, hidden_dim):
  """
  Function to calculate Discriminative score
  args: 
    - num_batches: How many batches of data should be evaluated? 
    - G: Trained Generator
    - D: Trained Discriminator
    - S: Trained Supervisor
    - E: Trained Embedder
    - dataloader: the dataloader used for training
    - batch_size: The batch size used for training
    - dim: the number of features in the data
    - seq_len: the length of each time series 
    - hidden_dim: the hidden dimension of the modules

  """
  # optimally one should split the data into training, test, and validation set for this step
  scores = []
  for i in range(num_batches):
    x = next(iter(dataloader))
    random_data = random_generator(batch_size=batch_size, z_dim=dim, 
                                 T_mb=extract_time(x)[0], max_seq_len=extract_time(x)[1])
        
    z = torch.tensor(random_data)
    z = z.float()
    # getting discriminator output for generated data    
    e_hat, _ = G(z)
    e_hat = torch.reshape(e_hat, (batch_size, seq_len, hidden_dim))
        
    H_hat, _ = Supervisor(e_hat)
    H_hat = torch.reshape(H_hat, (batch_size, seq_len, hidden_dim))
        
    Y_pred_fake = D(H_hat)
    # getting discriminator output for real data
    embed, _ = E(x)
    embed = torch.reshape(embed, (batch_size, seq_len, hidden_dim))

    Y_pred_real = D(embed)
    # calculate scores for batch
    bce = nn.BCEWithLogitsLoss()
    score_fake = bce(Y_pred_fake, torch.zeros_like(Y_pred_fake))
    score_real = bce(Y_pred_real, torch.ones_like(Y_pred_real))
    total = torch.add(score_real, score_fake)
    avg = torch.abs(total/2)
    avg = avg.detach().numpy()
    # append to scores
    scores.append(avg)
  
  # return average of scores for all batches
  return np.mean(scores)