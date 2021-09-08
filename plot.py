import matplotlib.pyplot as plt


def plot_error(S_error, C_error):
    
    fig = plt.figure(figsize=(15, 5))
    a=fig.add_subplot(1,2,1)
        
    imgplot = plt.plot(S_error)
    a.set_title('Errors in the secret image')
    
    a=fig.add_subplot(1,2,2)
    imgplot = plt.plot(C_error)
    a.set_title('Errors in the cover image.')
    
    plt.show()



# Number of secret and cover pairs to show.
def show_image(img, n_rows, n_col, idx, title=None):
    ax = plt.subplot(n_rows, n_col, idx)
    plt.imshow(img)
    #ax.get_xaxis().set_visible(False)
    #ax.get_yaxis().set_visible(False)
    ax.set_title(title)
