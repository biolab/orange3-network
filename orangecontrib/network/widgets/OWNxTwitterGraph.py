import twitter
import networkx as nx
from Orange.widgets import widget, gui
from Orange.data import Domain, StringVariable, Table
import numpy as np
from Orange.widgets.credentials import CredentialManager
import orangecontrib.network as network
from orangecontrib.text import twitter as txt_twitter
from PyQt4 import QtGui, QtCore
from collections import defaultdict

class OWNxTwitterGraph(widget.OWWidget):
    class APICredentialsDialog(widget.OWWidget):
        name = "Twitter API Credentials"
        want_main_area = False
        resizing_enabled = False

        cm_key = CredentialManager('Twitter API Key')
        cm_secret = CredentialManager('Twitter API Secret')

        key_input = ''
        secret_input = ''

        class Error(widget.OWWidget.Error):
            invalid_credentials = widget.Msg('These credentials are invalid.')

        def __init__(self, parent):
            super().__init__()
            self.parent = parent
            self.credentials = None

            form = QtGui.QFormLayout()
            form.setMargin(5)
            self.key_edit = gui.lineEdit(self, self, 'key_input', controlWidth=400)
            form.addRow('Key:', self.key_edit)
            self.secret_edit = gui.lineEdit(self, self, 'secret_input', controlWidth=400)
            form.addRow('Secret:', self.secret_edit)
            self.controlArea.layout().addLayout(form)

            self.submit_button = gui.button(self.controlArea, self, "OK", self.accept)

            self.load_credentials()

        def load_credentials(self):
            self.key_edit.setText(self.cm_key.key)
            self.secret_edit.setText(self.cm_secret.key)

        def save_credentials(self):
            self.cm_key.key = self.key_input
            self.cm_secret.key = self.secret_input

        def check_credentials(self):
            c = txt_twitter.Credentials(self.key_input, self.secret_input)
            if self.credentials != c:
                if c.valid:
                    self.save_credentials()
                else:
                    c = None
                self.credentials = c

        def accept(self, silent=False):
            if not silent: self.Error.invalid_credentials.clear()
            self.check_credentials()
            if self.credentials and self.credentials.valid:
                super().accept()
            elif not silent:
                self.Error.invalid_credentials()

    name = "Twitter User Graph"
    description = "Create a graph of Twitter users."
    icon = "icons/Twitter.svg"
    priority = 6470
    outputs = [("Followers", network.Graph),
               ("Following", network.Graph),
               ("All", network.Graph)]

    want_main_area = False


    def __init__(self):
        super().__init__()

        self.n_all = 0
        self.n_followers = 0
        self.n_following = 0
        self.on_rate_limit = None
        self.api_dlg = self.APICredentialsDialog(self)

        # GUI
        # Set API key button.
        key_dialog_button = gui.button(self.controlArea, self, 'Twitter API Key',
                                       callback=self.open_key_dialog,
                                       tooltip="Set the API key for this widget.")
        key_dialog_button.setFocusPolicy(QtCore.Qt.NoFocus)
        box = gui.widgetBox(self.controlArea, "Info")
        box.layout().addWidget(QtGui.QLabel("Users:"))
        self.users = QtGui.QTextEdit()
        box.layout().addWidget(self.users)
        gui.label(box, self, 'Following: %(n_following)d\n'
                             'Followers: %(n_followers)d\n'
                             'All: %(n_all)d')
        self.button = gui.button(box, self, "Create graph", self.fetch_users)

    def open_key_dialog(self):
        self.api_dlg.exec_()

    def fetch_users(self):
        CONSUMER_KEY = CredentialManager('Twitter API Key').key
        CONSUMER_SECRET = CredentialManager('Twitter API Secret').key
        OAUTH_TOKEN = ''
        OAUTH_TOKEN_SECRET = ''
        auth = twitter.oauth.OAuth(OAUTH_TOKEN, OAUTH_TOKEN_SECRET, CONSUMER_KEY, CONSUMER_SECRET)
        t = twitter.Twitter(auth=auth)

        followers_graph = nx.Graph()
        following_graph = nx.Graph()
        all_users = nx.Graph()
        users = self.users.toPlainText().split("\n")
        mapping = defaultdict(list)
        fwers_id = 0
        fwing_id = 0
        all_id = 0

        fwers_names=[]
        fwing_names=[]
        all_names=[]

        for user in users:
            result = t.users.show(screen_name=user)
            id = result["id"]

            if id in mapping.keys():
                user_fwer_id, user_fwing_id, user_all_id = mapping[id]
                if not user_fwer_id:
                    mapping[id][0] = fwers_id
                    user_fwer_id = fwers_id
                    fwers_id += 1
                    fwers_names.append([user, user])
                else:
                    fwers_names[user_fwer_id] = [user, user]
                if not user_fwing_id:
                    mapping[id][1] = fwing_id
                    user_fwing_id = fwing_id
                    fwing_id += 1
                    fwing_names.append([user, user])
                else:
                    fwing_names[user_fwing_id] = [user, user]
                all_names[user_all_id] = [user, user]
            else:
                mapping[id].extend([fwers_id, fwing_id, all_id])
                user_fwer_id, user_fwing_id, user_all_id = fwers_id, fwing_id, all_id
                fwers_id += 1
                fwing_id += 1
                all_id += 1
                fwers_names.append([user, user])
                fwing_names.append([user, user])
                all_names.append([user, user])
            followers_graph.add_node(user_fwer_id)
            following_graph.add_node(user_fwing_id)
            all_users.add_node(user_all_id)

            cursor = -1
            while cursor != 0:
                response = t.followers.ids(screen_name=user, cursor=cursor)
                for f_id in response['ids']:
                    if f_id in mapping.keys():
                        if not mapping[f_id][0]:
                            mapping[f_id][0] = fwers_id
                            fwers_names.append(["", name])
                            fwers_id += 1
                        followers_graph.add_edge(user_fwer_id, mapping[f_id][0])
                        all_users.add_edge(user_all_id, mapping[f_id][2])
                    else:
                        mapping[f_id].extend([fwers_id, None, all_id])
                        followers_graph.add_edge(user_fwer_id, fwers_id)
                        all_users.add_edge(user_all_id, all_id)
                        name = str(f_id)
                        fwers_names.append(["", name])
                        all_names.append(["", name])
                        fwers_id += 1
                        all_id += 1
                cursor = response['next_cursor']
            cursor = -1
            while cursor != 0:
                response = t.friends.ids(screen_name=user, cursor=cursor)
                for f_id in response['ids']:
                    if f_id in mapping.keys():
                        if not mapping[f_id][1]:
                            mapping[f_id][1] = fwing_id
                            fwing_names.append(["", name])
                            fwing_id += 1
                        following_graph.add_edge(user_fwing_id, mapping[f_id][1])
                        all_users.add_edge(user_all_id, mapping[f_id][2])
                    else:
                        mapping[f_id].extend([None, fwing_id, all_id])
                        following_graph.add_edge(user_fwing_id, fwing_id)
                        all_users.add_edge(user_all_id, all_id)
                        name = str(f_id)
                        fwing_names.append(["", name])
                        all_names.append(["", name])
                        fwing_id += 1
                        all_id += 1
                cursor = response['next_cursor']

        ids = list(set(mapping.keys()))
        for i in range(0, len(ids), 100):
            result = t.users.lookup(user_id=ids[i:i+100])
            for r in result:
                if mapping[r["id"]][0] :
                    fwers_names[mapping[r["id"]][0]][1] = r["screen_name"]
                if mapping[r["id"]][1] :
                    fwing_names[mapping[r["id"]][1]][1] = r["screen_name"]
                all_names[mapping[r["id"]][2]][1] = r["screen_name"]

        all_users = network.readwrite._wrap(all_users)
        followers = network.readwrite._wrap(followers_graph)
        following = network.readwrite._wrap(following_graph)

        meta_vars = [StringVariable('Starting users'), StringVariable('All users')]
        domain = Domain([], metas=meta_vars)

        table_fwers = Table.from_numpy(domain,
                                 np.array([[] for _ in range(len(fwers_names))]),
                                 metas=np.array(fwers_names, dtype=str))
        followers.set_items(table_fwers)
        table_fwing = Table.from_numpy(domain,
                                 np.array([[] for _ in range(len(fwing_names))]),
                                 metas=np.array(fwing_names, dtype=str))
        following.set_items(table_fwing)
        table_all = Table.from_numpy(domain,
                                 np.array([[] for _ in range(len(all_names))]),
                                 metas=np.array(all_names, dtype=str))
        all_users.set_items(table_all)

        self.send("Followers", followers)
        self.send("Following", following)
        self.send("All", all_users)

        self.n_all = len(all_users)
        self.n_followers = len(followers)
        self.n_following = len(following)

if __name__== "__main__":
    from PyQt4.QtGui import QApplication
    a = QApplication([])
    ow = OWNxTwitterGraph()
    ow.show()
    a.exec_()
