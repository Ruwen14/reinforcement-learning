import os
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense



class ActorCriticSharedNetwork(keras.Model):
    def __init__(self, n_actions, fc1_dims=1024, fc2_dims=512,
                     name='actor_critic', chkpt_dir='./checkpoints'):
        super(ActorCriticSharedNetwork, self).__init__()


        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.model_name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name + '_ac')

        # We share a common layer for both Actor and Critic.
        # As its input it takes the state. The State dimensionality
        # is affected by its represantation.
        # This is the first hidden layer
        self.fc1 = Dense(self.fc1_dims, activation='relu')
        #This is the second hidden layer
        self.fc2 = Dense(self.fc2_dims, activation='relu')

        # [CRITIC]
        # This layer is the state-value-function approximation
        # It outputs the estimated state-value, for a given state
        self.value_function = Dense(1, activation=None)


        # [ACTOR]
        # This layer is the for the policy.
        # Given a state, it outputs the probability distribution
        # over the n- output actions (n_actions)
        # softmax needed to normalize distribution from 0.0 to 1.0
        self.policy =Dense(n_actions, activation='softmax')

    # Forward function from pytorch
    def call(self, state):
        # The state is passed to the first hidden layer of
        # our shared network
        out1 = self.fc1(state)

        # Then it passed to the second hidden layer
        out2 = self.fc2(out1)

        # Through our shared parameters, we learn how to estimate
        # a state_value for a given state
        state_value = self.value_function(out2)

        # Through our shared parameters, we learn to make better
        # distributions of our deterministic actions to take
        policy_dist = self.policy(out2)

        return state_value, policy_dist


    def save_model(self):
        print(" Saving Model  ")
        self.save_weights(self.checkpoint_file)

    def load_model(self):
        print(" Loading Model")
        self.load_weights(self.checkpoint_file)









if __name__ == '__main__':
    pass


