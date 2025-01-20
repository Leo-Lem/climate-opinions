from __params__ import MODEL, CRAWL_PLATFORM

if CRAWL_PLATFORM:
    from src.crawl import twitter, youtube, bluesky

    if CRAWL_PLATFORM == "twitter":
        twitter()
    elif CRAWL_PLATFORM == "youtube":
        youtube()
    elif CRAWL_PLATFORM == "bluesky":
        bluesky()
else:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from src import preprocess, train, evaluate, predict, visualize

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL,
                                                               num_labels=3)

    training, validation, testing = preprocess(tokenizer)

    trainer = train(model, tokenizer, training, validation)
    results = evaluate(trainer, testing)
    print(results)

    predictions = predict("twitter")
    print(predictions.head())

    visualize("sample-twitter-predictions")
