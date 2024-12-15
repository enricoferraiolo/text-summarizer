import numpy as np

class InferenceUtils:
    @staticmethod
    def decode_sequence(encoder_model, decoder_model, input_seq, target_word_index, reverse_target_word_index, max_summary_len):
        e_out, e_h, e_c = encoder_model.predict(input_seq)

        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = target_word_index["sostok"]

        stop_condition = False
        decoded_sentence = ""
        while not stop_condition:
            output_tokens, h, c = decoder_model.predict([target_seq] + [e_out, e_h, e_c])
            sampled_token_index = np.argmax(output_tokens[0, -1, :])
            sampled_token = reverse_target_word_index[sampled_token_index]

            if sampled_token != "eostok":
                decoded_sentence += " " + sampled_token

            if sampled_token == "eostok" or len(decoded_sentence.split()) >= (max_summary_len - 1):
                stop_condition = True

            target_seq = np.zeros((1, 1))
            target_seq[0, 0] = sampled_token_index

            e_h, e_c = h, c

        return decoded_sentence

    @staticmethod
    def seq2summary(input_seq, target_word_index, reverse_target_word_index):
        newString = ""
        for i in input_seq:
            if i != 0 and i != target_word_index["sostok"] and i != target_word_index["eostok"]:
                newString += reverse_target_word_index[i] + " "
        return newString.strip()

    @staticmethod
    def seq2text(input_seq, reverse_source_word_index):
        newString = ""
        for i in input_seq:
            if i != 0:
                newString += reverse_source_word_index[i] + " "
        return newString.strip()
