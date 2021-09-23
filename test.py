import OpenAttack
scpn = OpenAttack.attackers.SCPNAttacker()
sent = "I love you"
paraphrases = scpn.gen_paraphrase(sent, scpn.templates)