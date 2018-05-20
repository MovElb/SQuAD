import config
import telebot
from prepro import annotate, to_id, init
from model_init import init_model
from train import BatchGen
import argparse
from drqa.utils import str2bool
import torch
import traceback


def is_ascii(text):
    return all(ord(c) < 128 for c in text)


class QA:
    S_WAIT_CONTEXT = 0
    S_WAIT_QUESTION = 1
    S_ANSWERING = 2
    S_ANSWERED = 3
    S_NULL = 4

    def __init__(self):
        self.contexts = {}
        self.questions = {}
        self.state = {}


    def add_user(self, id):
        self.state[id] = self.S_WAIT_CONTEXT


    def add_context(self, id, context):
        self.state[id] = self.S_WAIT_QUESTION
        self.contexts[id] = context


    def add_question(self, id, question):
        self.state[id] = self.S_ANSWERING
        self.questions[id] = question


    def get_pair(self, id):
        if (not self.S_ANSWERING):
            raise IOError

        return self.contexts[id], self.questions[id]


    def get_state(self, id):
        return self.state.get(id, self.S_NULL)


    def set_null(self, id):
        self.state[id] = self.S_NULL


    def set_answering(self, id):
        self.state[id] = self.S_ANSWERING


qa = QA()
bot = telebot.TeleBot(config.token)
counter = 0


@bot.message_handler(commands=['start'])
def start(message):
    bot.send_message(message.chat.id, 'Hello, I\'m a bot who can answer to your questions. '
                                      'Send /ask command to start.')


@bot.message_handler(commands=['ask'])
def asking(message):
    qa.add_user(message.chat.id)
    bot.send_message(message.chat.id, 'Send me context.')


@bot.message_handler(func=lambda msg: qa.get_state(msg.chat.id) == qa.S_WAIT_CONTEXT, content_types=['text'])
def get_context(message):
    if (not is_ascii(message.text)):
        bot.send_message(message.chat.id, 'Your text contains non-ASCII symbols, try again.')
        return

    qa.add_context(message.chat.id, message.text)
    bot.send_message(message.chat.id, 'Now send me a question, which\'s answer is in the text.')


@bot.message_handler(func=lambda msg: qa.get_state(msg.chat.id) == qa.S_WAIT_QUESTION, content_types=['text'])
def get_question(message):
    if (not is_ascii(message.text)):
        bot.send_message(message.chat.id, 'Your question contains non-ASCII symbols, try again.')
        return

    qa.add_question(message.chat.id, message.text)

    pair = qa.get_pair(message.chat.id)
    qa.set_answering(message.chat.id)

    global counter
    counter += 1

    try:
        annotated = annotate(('interact-{}'.format(counter), pair[0], pair[1]), meta['wv_cased'])
        model_in = to_id(annotated, w2id, tag2id, ent2id)
        model_in = next(iter(BatchGen([model_in], batch_size=1, gpu=args.cuda, evaluation=True)))
        prediction = model.predict(model_in)[0]

        bot.send_message(message.chat.id, "Answer:")
        bot.send_message(message.chat.id, prediction)
        qa.set_null(message.chat.id)
    except Exception as e:
        traceback.print_exc()
        bot.send_message(message.chat.id, "Something went wrong, try again with /ask command.")
        qa.set_null(message.chat.id)


@bot.message_handler(content_types=['text'])
def answer_default(message):
    bot.send_message(message.chat.id, "Send /ask to start.")


parser = argparse.ArgumentParser(
    description='Interact with document reader model.'
)
parser.add_argument('--model-file', default='models/best_model.pt',
                    help='path to model file')
parser.add_argument("--cuda", type=str2bool, nargs='?',
                    const=True, default=torch.cuda.is_available(),
                    help='whether to use GPU acceleration.')


if __name__ == '__main__':
    args = parser.parse_args()

    meta, w2id, tag2id, ent2id, model = init_model(args)
    print("Model is loaded.")
    bot.polling(none_stop=True)
