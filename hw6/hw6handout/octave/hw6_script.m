% Point to data directory here
% By default, we are pointing to '../data/'
data_dir = ['..', filesep(), 'data'];

% Uncomment the next line if csv2cell function is not installed
%pkg install -forge io
pkg load io

% Load data files
X = csvread([data_dir, filesep(), 'kmeans_test_data.csv']);

% TODO: Test update_assignments function, defined in update_assignments.m

% TODO: Test update_centers function, defined in update_centers.m

% TODO: Test lloyd_iteration function, defined in lloyd_iteration.m

% TODO: Test kmeans_obj function, defined in kmeans_obj.m

% TODO: Run experiments outlined in HW6 PDF

% For question 9 and 10
mnist_X = csvread([data_dir, filesep(), 'mnist_data.csv']);