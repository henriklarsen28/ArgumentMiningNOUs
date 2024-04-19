from transformers import pipeline
from pandas import DataFrame
from src.util.helpers import select_device

classifier = pipeline('text-classification', model="NicolaiSivesind/NOU-Classifier-Raw", device="mps")

texts = ["Klimautvalget mener kraftprisene må reflektere at kraft er en knapp ressurs for å gi insentiver til å "
         " begrense bruken av kraft. Treindustrien mener det er behov for virkemidler utover prissignalet for å fremme "
         " energieffektivisering. Tilgang på rimelig, fornybar kraft har vært en viktig rammebetingelse og "
         " konkurransefortrinn for norsk industri, og er en forutsetning for videre grønn industriutvikling. Det er "
         " nødvendig med politikkutforming som ivaretar også dette på både kort og lang sikt.",
         """Med relevans for klimautslipp omfatter matsystemet bl.a. arealbruk og -bearbeiding, innsatsvarer, logistikk, prosessering av varer og tilgjengelighet/sporbarhet av dokumentasjon gjennom hele verdikjedene. Dette gjelder også for importerte matvarer. Sporbarhet er avgjørende både for næringsaktørene og myndighetene med tanke på å kunne rapportere ift. erklærte mål om reduserte utslipp eller klimapositivitet. Matkriminalitet som f.eks. tilsiktet feilmerking av produkter, er en utbredt form for svindel. I et system som kan forsterke økonomiske insitamenter for slik svindel – knyttet til uriktige data om klimapåvirkning, vil det i alvorlig grad kunne undergrave lavutslippsmålet. Krav om tilgang til og kvaliteten på data som skal innarbeides i sporbarhetssystemet må derfor ha høy prioritet. Som bidrag til å stimulere alle samfunnsaktører til å ta klimapositive valg er det også viktig at dokumentasjonen gjøres tilgjengelig og forståelig for vanlige samfunnsborgere. Merkeordninger og/eller andre sporbarhetsbaserte systemer som er nøytrale, informative og brukervennlige bør derfor prioriteres og knyttes opp mot kvalitetssikrete beregningsmodeller og det oppdaterte kunnskapsgrunnlaget."""]

result = classifier(texts)

print(result)


class Classifier:
    def __init__(self, model: str, test_df):
        self.model = model
        self.test_df = test_df
        self.pipe = pipeline('text-classification',
                             model="NicolaiSivesind/NOU-Classifier-Raw",
                             device=select_device())
