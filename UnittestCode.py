import unittest
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam

class TestTrajectoryModel(unittest.TestCase):

    def setUp(self):
        # Load test data or create test data if necessary
        df = pd.read_csv('combined_traj_data.csv')
        df['speed'] = np.sqrt(df['vx'] ** 2 + df['vy'] ** 2)
        df['heading'] = np.arctan2(df['vy'], df['vx'])
        df['accel'] = df['speed'].diff() / df['t'].diff()
        df['turn_rate'] = df['heading'].diff() / df['t'].diff()
        df['distance'] = np.sqrt(df['x'].diff() ** 2 + df['y'].diff() ** 2)
        df['accel'].fillna(0, inplace=True)
        df['turn_rate'].fillna(0, inplace=True)
        df['distance'].fillna(0, inplace=True)
        features = df[['speed', 'heading', 'accel', 'turn_rate', 'distance']]
        self.test_features = features
        self.test_data = df

    def test_feature_engineering(self):
        for i in range(len(self.test_data)):
            row = self.test_data.iloc[i]
            expected_speed = np.sqrt(row['vx'] ** 2 + row['vy'] ** 2)
            expected_accel = 0 if i == 0 else (row['speed'] - self.test_data['speed'].iloc[i - 1]) / (
                        row['t'] - self.test_data['t'].iloc[i - 1])
            expected_turn_rate = 0 if i == 0 else (row['heading'] - self.test_data['heading'].iloc[i - 1]) / (
                        row['t'] - self.test_data['t'].iloc[i - 1])
            expected_distance = 0 if i == 0 else np.sqrt(
                (row['x'] - self.test_data['x'].iloc[i - 1]) ** 2 + (row['y'] - self.test_data['y'].iloc[i - 1]) ** 2)

            # Check if the calculated values match the expected values
            self.assertAlmostEqual(row['speed'], expected_speed, places=2)
            self.assertAlmostEqual(row['accel'], expected_accel, places=2)
            self.assertAlmostEqual(row['turn_rate'], expected_turn_rate, places=2)
            self.assertAlmostEqual(row['distance'], expected_distance, places=2)

    def test_outlier_removal(self):
        # Generate a simple mock feature dataset for outlier removal
        mock_features = np.random.rand(100, 5)  # 100 samples, 5 features

        # Create a Local Outlier Factor (LOF) model
        lof = LocalOutlierFactor()

        # Fit the LOF model to the mock features and predict outliers
        outlier_predictions = lof.fit_predict(mock_features)

        print(outlier_predictions)
        # Check if the predictions contain no -1 (indicating outliers)
        self.assertFalse(-1 in outlier_predictions)

    def test_clustering(self):

        kmeans = KMeans(n_clusters=5)

        # Fit the model to the mock features
        cluster_labels = kmeans.fit_predict(self.test_features)

        # Check if the number of unique cluster labels is as expected
        self.assertEqual(len(np.unique(cluster_labels)), 5)

    def test_model_training(self):
        lof = LocalOutlierFactor()
        y_pr = lof.fit_predict(self.test_features)
        inlier_indices = y_pr == 1
        features_no_outliers = self.test_features[inlier_indices]
        features_no_outliers = features_no_outliers.reset_index(drop=True)
        # Cluster trajectories
        kmeans = KMeans(n_clusters=5)
        features_no_outliers['cluster'] = kmeans.fit_predict(features_no_outliers)
        # Identify normal cluster
        cluster_list = features_no_outliers['cluster'].tolist()
        normal_cluster = max(set(cluster_list), key=cluster_list.count)
        features_no_outliers['cluster'] = np.where(features_no_outliers['cluster'] == normal_cluster, 1, 0)
        normal_features = features_no_outliers[['speed', 'heading', 'accel', 'turn_rate', 'distance']]
        X = normal_features.values
        y = features_no_outliers['cluster'].values
        y_labelencoder = LabelEncoder()
        Y = y_labelencoder.fit_transform(y)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
        NUM_CLASSES = len(np.unique(Y_train))
        y_train = tf.one_hot(Y_train, NUM_CLASSES)
        y_test = tf.one_hot(Y_test, NUM_CLASSES)
        batch_size = 32
        # Reshape the input data to match the expected shape for Conv1D
        x_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        x_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

        # Create the Sequential model
        model = Sequential()
        # First Conv1D layer with 32 filters, kernel size of 3, and 'relu' activation
        model.add(Conv1D(32, kernel_size=3, activation='relu', input_shape=(x_train.shape[1], x_train.shape[2])))
        # BatchNormalization
        model.add(BatchNormalization())
        # MaxPooling
        model.add(MaxPooling1D(pool_size=1))
        # Dropout
        model.add(Dropout(0.25))
        # Second Conv1D layer with 64 filters and a kernel size of 3
        model.add(Conv1D(64, kernel_size=3, activation='relu'))
        # BatchNormalization
        model.add(BatchNormalization())
        # MaxPooling
        model.add(MaxPooling1D(pool_size=1))
        # Dropout
        model.add(Dropout(0.25))
        # Flatten layer to convert the output to a 1D vector
        model.add(Flatten())
        # Fully connected layers
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(32, activation='relu'))
        model.add(Dropout(0.2))
        # Output layer with appropriate activation (e.g., sigmoid for binary classification)
        model.add(Dense(2, activation='softmax'))
        # Compile the model
        model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])
        # Print the model summary to check the shapes
        model.summary()
        epochs = 1
        # Train the model
        history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,
                                      validation_data=(x_test, y_test))
        # Check if training accuracy is above a certain threshold (e.g., 0.7)
        self.assertTrue(history.history['val_accuracy'][-1] > 0.7)


    def test_prediction(self):
        lof = LocalOutlierFactor()
        y_pr = lof.fit_predict(self.test_features)
        inlier_indices = y_pr == 1
        features_no_outliers = self.test_features[inlier_indices]
        features_no_outliers = features_no_outliers.reset_index(drop=True)
        # Cluster trajectories
        kmeans = KMeans(n_clusters=5)
        features_no_outliers['cluster'] = kmeans.fit_predict(features_no_outliers)

        # Identify normal cluster
        cluster_list = features_no_outliers['cluster'].tolist()
        normal_cluster = max(set(cluster_list), key=cluster_list.count)
        features_no_outliers['cluster'] = np.where(features_no_outliers['cluster'] == normal_cluster, 1, 0)
        normal_features = features_no_outliers[['speed', 'heading', 'accel', 'turn_rate', 'distance']]
        X = normal_features.values
        y = features_no_outliers['cluster'].values
        y_labelencoder = LabelEncoder()
        Y = y_labelencoder.fit_transform(y)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
        NUM_CLASSES = len(np.unique(Y_train))
        y_train = tf.one_hot(Y_train, NUM_CLASSES)
        y_test = tf.one_hot(Y_test, NUM_CLASSES)
        batch_size = 32

        # Reshape the input data to match the expected shape for Conv1D
        x_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        x_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
        # Create the Sequential model
        model = Sequential()

        # First Conv1D layer with 32 filters, kernel size of 3, and 'relu' activation
        model.add(Conv1D(32, kernel_size=3, activation='relu', input_shape=(x_train.shape[1], x_train.shape[2])))

        # BatchNormalization
        model.add(BatchNormalization())

        # MaxPooling
        model.add(MaxPooling1D(pool_size=1))

        # Dropout
        model.add(Dropout(0.25))

        # Second Conv1D layer with 64 filters and a kernel size of 3
        model.add(Conv1D(64, kernel_size=3, activation='relu'))

        # BatchNormalization
        model.add(BatchNormalization())

        # MaxPooling
        model.add(MaxPooling1D(pool_size=1))

        # Dropout
        model.add(Dropout(0.25))

        # Flatten layer to convert the output to a 1D vector
        model.add(Flatten())

        # Fully connected layers
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.2))

        model.add(Dense(32, activation='relu'))
        model.add(Dropout(0.2))

        # Output layer with appropriate activation (e.g., sigmoid for binary classification)
        model.add(Dense(2, activation='softmax'))

        # Compile the model
        model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

        # Print the model summary to check the shapes
        model.summary()

        epochs = 1

        # Train the model
        history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test))

        # Generate 15 normal trajectories
        num_traj = 15
        traj_len = 50
        normal_trajectories = []
        for i in range(num_traj):
            x = np.random.uniform(low=-5.0, high=5.0, size=traj_len)
            y = np.random.uniform(low=40.0, high=60.0, size=traj_len)
            vx = np.random.normal(loc=10, scale=2, size=traj_len)
            vy = np.random.normal(loc=10, scale=2, size=traj_len)
            t = np.arange(traj_len)
            traj_num = np.full(traj_len, i + 1)
            trajectory = np.column_stack((x, y, vx, vy, t, traj_num))
            normal_trajectories.append(trajectory)

        # Generate location-based anomalies
        num_loc_anom = 7
        loc_anom_traj = []
        for i in range(num_loc_anom):
            t = np.random.randint(10, 40)
            x = np.random.uniform(low=10, high=15)
            y = np.random.uniform(low=10, high=15)
            anomaly_traj = normal_trajectories[i].copy()
            anomaly_traj[t, 0] = x
            anomaly_traj[t, 1] = y
            loc_anom_traj.append(anomaly_traj)

        # Generate velocity-based anomalies
        num_vel_anom = 8
        vel_anom_traj = []
        for i in range(num_vel_anom):
            t = np.random.randint(10, 40)
            vx = np.random.normal(loc=40, scale=5)
            vy = np.random.normal(loc=40, scale=5)
            anomaly_traj = normal_trajectories[i + num_loc_anom].copy()
            anomaly_traj[t, 2] = vx
            anomaly_traj[t, 3] = vy
            vel_anom_traj.append(anomaly_traj)

        abnormal_trajectories = loc_anom_traj + vel_anom_traj

        # List of column names
        cols = ['x', 'y', 'vx', 'vy', 't', 'traj_number']

        # Create empty dataframe
        df = pd.DataFrame(columns=cols)

        # Loop through trajectories and append to dataframe
        for traj in abnormal_trajectories:
            df_temp = pd.DataFrame(traj, columns=cols)
            df = pd.concat([df, df_temp])

        df.reset_index(drop=True)
        # Feature engineering
        df['speed'] = np.sqrt(df['vx'] ** 2 + df['vy'] ** 2)
        df['heading'] = np.arctan2(df['vy'], df['vx'])
        df['accel'] = df['speed'].diff() / df['t'].diff()
        df['turn_rate'] = df['heading'].diff() / df['t'].diff()
        df['distance'] = np.sqrt(df['x'].diff() ** 2 + df['y'].diff() ** 2)
        features = df[['speed', 'heading', 'accel', 'turn_rate', 'distance']]
        features['accel'].fillna(0, inplace=True)
        features['turn_rate'].fillna(0, inplace=True)
        features['distance'].fillna(0, inplace=True)
        features['cluster'] = kmeans.predict(features)
        features['cluster'] = np.where(features['cluster'] == normal_cluster, 1, 0)
        X_val = features[['speed', 'heading', 'accel', 'turn_rate', 'distance']]
        x_val = X_val.values
        y_val = features['cluster'].values
        y_validation = y_labelencoder.fit_transform(y_val)
        NUM_CLASSES = len(np.unique(y_validation))
        y_valid = tf.one_hot(y_validation, NUM_CLASSES)
        x_valid = x_val.reshape(x_val.shape[0], x_val.shape[1], 1)
        y_pred = model.predict(x_valid)

        # Assuming y_pred contains probability scores for each class, you can convert them to class labels
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = np.argmax(y_valid, axis=1)

        # Calculate the accuracy of the predictions
        accuracy = accuracy_score(y_true_classes, y_pred_classes)
        print(accuracy)
        self.assertTrue(accuracy > 0.7)


if __name__ == '__main__':
    unittest.main()
